from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain_community.retrievers import PubMedRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import requests, json
import os
from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

def documentloader_fromprompt(research_topic, top_k):
    print(f"Debug: documentloader_fromprompt called with research_topic='{research_topic}' and top_k='{top_k}'")
    try:    
        protocol = "https"
        hostname = "chat.cosy.bio"


        host = f"{protocol}://{hostname}"
        
        account = {'email':'melanie.altmann@studium.uni-hamburg.de', 'password':'be$oPe96'}
        #account = {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
        auth_url = f"{host}/api/v1/auths/signin"
        api_url = f"{host}/ollama"

        auth_response = requests.post(auth_url, json=account)
        jwt= json.loads(auth_response.text)["token"]

        llm = Ollama(base_url=api_url, model="llama3", temperature=0.0, headers= {"Authorization": "Bearer " + jwt})

        embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text", headers= {"Authorization": "Bearer " + jwt})

        retriever = PubMedRetriever(top_k_results=top_k)

        prompt = research_topic
        docs = retriever.invoke(prompt)
        
        if not docs:
            print("Debug: No documents found")
            return None

        print(f"Debug: Loaded {len(docs)} documents")
        return docs
        
    except Exception as e:
        print(f"Error in documentloader_fromprompt: {str(e)}")
        return None

def rephrasing_input_to_fit_pubmedapiwrapper(research_topic):
    try:    
        protocol = "https"
        hostname = "chat.cosy.bio"


        host = f"{protocol}://{hostname}"
        
        account = {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
        auth_url = f"{host}/api/v1/auths/signin"
        api_url = f"{host}/ollama"

        auth_response = requests.post(auth_url, json=account)
        jwt= json.loads(auth_response.text)["token"]

        llm = Ollama(base_url=api_url, model="llama3", temperature=0.0, headers= {"Authorization": "Bearer " + jwt})

        template = "Rephrase this text for a PubMed search query: {text}. Answer with only your rephrased query."
        prompt = PromptTemplate(template=template, input_variables=["text"])

        chain = LLMChain(llm=llm, prompt = prompt)
        rephrased_query = chain.run(text=research_topic)
        print(f"Debug: Rephrased Query - {rephrased_query}")
        if rephrased_query is None:
            raise ValueError("Rephrasing returned None.")
        
        df = pd.DataFrame([[rephrased_query]], columns=['rephrased_query'])
        df.to_csv("txt.csv", index=False)
        print("Debug: txt.csv saved successfully.")
        return rephrased_query
        
    except Exception as e:
        print(f"Error while rephrasing: {str(e)}")
        return None
