from flask import Flask, render_template, request
from src.model import embed_model
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import pinecone
from langchain_pinecone.vectorstores import Pinecone
from langchain.llms import CTransformers
import os
import warnings
warnings.filterwarnings("ignore")
from src.prompt import *


app = Flask(__name__)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

embed = embed_model()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
docsearch = Pinecone.from_existing_index(PINECONE_INDEX, embed)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={"max_new_tokens": 512,
                            "temperature": 0.8})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa.invoke({'query': input})
    print("Response: ", result)
    return str(result["result"])

if __name__ == '__main__':
    app.run(debug=True)

