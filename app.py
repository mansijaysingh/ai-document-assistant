import os
from flask import Flask,render_template,request
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

UPLOAD_FOLDER='uploads'
if not os.path.exists(UPLOAD_FOLDER):
  os.mkdir(UPLOAD_FOLDER)


llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)

vector_store=None

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload_file():
  if 'pdf_file' in request.files:
    file=request.files['pdf_file']
    if file.filename !='':
      filepath=os.path.join(UPLOAD_FOLDER, file.filename)
      file.save(filepath)

      message="File uploaded and processed successfully."
      return render_template("index.html", message=message)
    