from langchain_community.llms import Ollama
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain_community.retrievers import PubMedRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests, json
import pickle as pkl

protocol = "https"
hostname = "chat.cosy.bio"


host = f"{protocol}://{hostname}"

account = {'email': 'melanie.altmann@studium.uni-hamburg.de', 'password': 'be$oPe96'}
auth_url = f"{host}/api/v1/auths/signin"
api_url = f"{host}/ollama"

auth_response = requests.post(auth_url, json=account)
jwt= json.loads(auth_response.text)["token"]

llm = Ollama(model="llama3", temperature=0.0, headers= {"Authorization": "Bearer " + jwt})
#embeddings = OllamaEmbeddings(model="nomic-embed-text")


retriever = PubMedRetriever()

# retriever.get_relevant_documents("chatgpt")

PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}
You are allowed to rephrase the answer based on the context. 
Question: {question}
"""
PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

# tag::qa[]
pubmed_qa = RetrievalQA.from_chain_type(
    llm,                  # <1>
    chain_type="stuff",   # <2>
    retriever=retriever,  # <3>
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

prompt = "Llamas are members of the camelid family."
#response = pubmed_qa({"query": prompt})
response = pubmed_qa.invoke(prompt)
print(response)
#response.to_pickle("response.pkl")