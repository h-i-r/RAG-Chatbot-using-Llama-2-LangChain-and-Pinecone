import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()

NASA_API_KEY = os.environ.get('NASA_API_KEY')
NASA_URL = os.environ.get('NASA_URL')
def load_responses():
    loader = WebBaseLoader(NASA_URL+NASA_API_KEY)
    documents = loader.load()
    return documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

def embed_model():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding


