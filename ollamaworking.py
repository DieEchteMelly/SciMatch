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
        
        account = {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
        auth_url = f"{host}/api/v1/auths/signin"
        api_url = f"{host}/ollama"

        auth_response = requests.post(auth_url, json=account)
        jwt= json.loads(auth_response.text)["token"]

        llm = Ollama(base_url=api_url, model="llama3", temperature=0.0, headers= {"Authorization": "Bearer " + jwt})

        embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text", headers= {"Authorization": "Bearer " + jwt})

        retriever = PubMedRetriever(top_k_results=top_k, sorting="Best_match")

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

        template = f"""As an expert in PubMed queries and drug repositioning research, your task is to generate a comprehensive PubMed search query for this topic: {research_topic}

        ## Guidelines:
        1. Identify 3-5 key concepts related to the topic.
        2. Use AND to combine main concepts, OR for synonyms within concepts.
        3. Use asterisk (*) for truncation, question mark (?) for wildcards.
        4. Use quotation marks for exact phrases.
        5. Use [Title/Abstract] for free-text searches, [Mesh] for MeSH terms (in quotes, lowercase).
        6. Use [All Fields] when appropriate for broader searches.
        7. Group terms for each concept in parentheses, combine with AND.
        8. Include relevant synonyms and related terms for key concepts.
        9. Balance between specificity (precision) and sensitivity (recall).
        10. Consider using NOT to exclude irrelevant results if necessary.

        ## Query Structure:
        ```
        ((Concept1[Title/Abstract] OR "Concept1"[Mesh] OR Synonym1[All Fields])
        AND
        (Concept2[Title/Abstract] OR "Concept2"[Mesh] OR Synonym2[All Fields])
        AND
        (Concept3[Title/Abstract] OR "Concept3"[Mesh] OR Synonym3[All Fields])
        AND
        (Concept4[Title/Abstract] OR "Concept4"[Mesh] OR Synonym4[All Fields]))
        ```

        ## Additional Considerations:
        - If the topic is related to reviews, consider including terms for different types of reviews (e.g., "systematic review"[Title/Abstract], "meta-analysis"[Title/Abstract]).
        - If the topic is about tools or software, include terms related to their availability (e.g., "software"[Title/Abstract], "tool*"[Title/Abstract], "github"[All Fields]).
        - If the topic involves databases, include terms related to database accessibility (e.g., "database"[Title/Abstract], "dataset"[Title/Abstract], "API"[All Fields]).

        Provide ONLY the query, no explanations. Start with (( and end with ))."""
        
        prompt = PromptTemplate(template=template, input_variables=["text"])

        chain = LLMChain(llm=llm, prompt = prompt)
        rephrased_query = chain.run(text=research_topic)
        print(f"Debug: Rephrased Query - {rephrased_query}")
        if rephrased_query is None:
            raise ValueError("Rephrasing returned None.")
        return rephrased_query
        
    except Exception as e:
        print(f"Error while rephrasing: {str(e)}")
        return None
