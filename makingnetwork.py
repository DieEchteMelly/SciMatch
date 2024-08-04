import pandas as pd
from langchain_community.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
import requests, json
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import normalize
from pyvis.network import Network
import seaborn as sns
from sklearn.cluster import DBSCAN
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

pd.set_option('display.max_colwidth', None)

def lets_embed(pmid_authors_df, parsedInfo, paper_main_authors):
    print("Debug: Starting lets_embed function")
    authors_df = pd.merge(pmid_authors_df, parsedInfo, left_on='pubmedid', right_on='uid', how='inner')
    authors_df['research'] = authors_df.apply(lambda row: str(row['Title']) + ' ' + str(row['page_content']), axis=1)
    protocol = "https"
    hostname = "chat.cosy.bio"
    host = f"{protocol}://{hostname}"

    account = {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
    auth_url = f"{host}/api/v1/auths/signin"
    api_url = f"{host}/ollama"

    auth_response = requests.post(auth_url, json=account)
    print(f"Debug: auth_response status code = {auth_response.status_code}")
    print(f"Debug: auth_response content = {auth_response.content}")

    if auth_response.status_code != 200:
        print(f"Error: Failed to authenticate. Status code: {auth_response.status_code}")
        return None

    try:
        jwt = auth_response.json().get("token")
        print(f"Debug: JWT token = {jwt}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {str(e)}")
        print(f"Response content: {auth_response.content}")
        return None

    if not jwt:
        print("Error: Failed to retrieve JWT token")
        return None

    headers = {"Authorization": "Bearer " + jwt}

    ollama_llm = Ollama(base_url=api_url, model="llama3", temperature=0.0, headers=headers)
    ollama_embeddings = OllamaEmbeddings(base_url=api_url, model='nomic-embed-text', headers=headers)

    loader = DataFrameLoader(authors_df, page_content_column="research")

    try:
        vector_db = FAISS.from_documents(loader.load(), ollama_embeddings)
        vector_db.save_local("test_net_embedding")
        vectors = vector_db.index.reconstruct_n(0, vector_db.index.ntotal)
    except Exception as e:
        print(f"Error during FAISS operations: {str(e)}")
        return None

    similarity_matrix = cosine_similarity(vectors)
    similarity_df = pd.DataFrame(similarity_matrix, index=authors_df.index, columns=authors_df.index)

    distance_matrix = cosine_distances(vectors)
    dbscan = DBSCAN(metric='precomputed', eps=0.1, min_samples=2)
    dbscan.fit(distance_matrix)
    cluster_labels = dbscan.labels_
    authors_df['Cluster'] = cluster_labels

    similarity_df = similarity_df.astype('float64')

    num_clusters = authors_df['Cluster'].nunique()
    palette = sns.color_palette("Spectral", num_clusters)
    cluster_colors = {cluster: f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                      for cluster, color in zip(authors_df['Cluster'].unique(), palette)}

    net = Network(height="750px", width="100%", bgcolor="#dddddd", font_color="black", filter_menu=True, notebook=True, cdn_resources='remote')
    net.toggle_hide_edges_on_drag(False)
    net.barnes_hut()

    highlight_authors = set(paper_main_authors['first_author']).union(set(paper_main_authors['last_author']))

    for idx, row in authors_df.iterrows():
        node = idx
        full_name = f"{row['forename']} {row['lastname']}"
        title = f"{full_name}: {row['affiliation']}"
        color = cluster_colors[row['Cluster']]
        size = 50 if full_name in highlight_authors else 35
        net.add_node(node, label=full_name, title=title, color=color, size=size)

    edge_color = '#a6a6a6'

    for i, source in enumerate(similarity_df.index):
        for j, target in enumerate(similarity_df.columns):
            if i < j:
                similarity = similarity_df.loc[source, target]
                if similarity >= 0.999:
                    source_name = f"{authors_df.loc[source, 'forename']} {authors_df.loc[source, 'lastname']}"
                    target_name = f"{authors_df.loc[target, 'forename']} {authors_df.loc[target, 'lastname']}"
                    edge_title = f"Edge from {source_name} to {target_name}. Similarity: {similarity:.2f}"
                    net.add_edge(source, target, title=edge_title, value=similarity, color=edge_color)

    print("Debug: Successfully created the network")
    return net