import os
from flask import Flask,render_template,request,session
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app=Flask(__name__)

app.config['SECRET_KEY']=os.urandom(24)


UPLOAD_FOLDER='uploads'
if not os.path.exists(UPLOAD_FOLDER):
  os.mkdir(UPLOAD_FOLDER)

vector_store=None
retrieval_chain=None




vector_store=None

@app.route('/')
def home():
  session.clear()
  return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload_file():
  global vector_store,retrieval_chain
  if 'pdf_file' in request.files:
    file=request.files['pdf_file']
    if file.filename !='':
      filepath=os.path.join(UPLOAD_FOLDER, file.filename)
      file.save(filepath)

      loader=PyPDFLoader(filepath)
      documents=loader.load()

      text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
      chunks=text_splitter.split_documents(documents)

      embeddings=OpenAIEmbeddings()
      vector_store=FAISS.from_documents(chunks, embeddings)

      llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)
      prompt=ChatPromptTemplate.from_template(
        """
                Answer the user's question based on the context provided.
                If you don't know the answer, just say that you don't know.
                
                <context>
                {context}
                </context>
                
                Question: {input}
                """
      )
      document_chain=create_stuff_documents_chain(llm,prompt)
      retriever=vector_store.as_retriever()
      retrieval_chain=create_retrieval_chain(retriever,document_chain)

      session['chat_history']=[]

      return render_template("index.html", message="PDF processed. You can now ask questions.")


  

   
    return render_template("index.html", message="File upload failed.") 
  
@app.route('/ask', methods=['POST'])
def ask_question():
  if retrieval_chain is None:
    return render_template("index.html", message="Please upload a PDF file first.")
  question=request.form.get('question')
  if not question:
    return render_template("index.html", chat_history=session.get('chat_history'))
  response=retrieval_chain.invoke({'input':question, 'chat_history': session.get('chat_history', [])})

  chat_history = session.get('chat_history', [])
  chat_history.append({"type": "user", "text": question})
  chat_history.append({"type": "bot", "text": response['answer']})
  session['chat_history'] = chat_history

  return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
    