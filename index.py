from src.model import load_responses, text_split, embed_model
from dotenv import load_dotenv
import pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore
import os
import warnings
from pinecone import ServerlessSpec
warnings.filterwarnings("ignore")
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

extracted_data = load_responses()
extracted_chunks = text_split(extracted_data)
embedding = embed_model()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX)
docs_chunks = [t.page_content for t in extracted_chunks]
docsearch = PineconeVectorStore.from_texts(texts=docs_chunks, embedding=embedding, index_name=PINECONE_INDEX)
