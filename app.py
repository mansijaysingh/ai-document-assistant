from flask import Flask, render_template
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')