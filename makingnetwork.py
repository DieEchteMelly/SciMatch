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
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
import os
from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

pd.set_option('display.max_colwidth',None)

def lets_embed(pmid_authors_df, parsedInfo, paper_main_authors):
    authors_df = pd.merge(pmid_authors_df, parsedInfo, left_on='pubmedid', right_on='uid', how='inner')
    authors_df['research'] = authors_df.apply(lambda row: str(row['Title'])+ ' ' + str(row['page_content']), axis=1)
    protocol = "https"
    hostname = "chat.cosy.bio"
    host = f"{protocol}://{hostname}"
    
    account = {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
    auth_url = f"{host}/api/v1/auths/signin"
    api_url = f"{host}/ollama"

    auth_response = requests.post(auth_url, json=account)
    jwt= json.loads(auth_response.text)["token"]

    ollama_llm = Ollama(base_url=api_url, model="llama3", temperature = 0.0, headers= {"Authorization": "Bearer " + jwt})
    ollama_embeddings = OllamaEmbeddings(base_url=api_url, model='nomic-embed-text', headers= {"Authorization": "Bearer " + jwt})

    loader = DataFrameLoader(authors_df, page_content_column="research")
    #documents=loader.load()
    #vectorstore = FAISS.from_documents(
    #    documents,embedding=ollama_embeddings,
    #)

    #Perform a similarity search
    #query = authors_df["uid"]
    #results = vectorstore.similarity_search_with_score(query)
    vector_db = FAISS.from_documents(loader.load(), ollama_embeddings)
    vector_db.save_local("test_net_embedding")
    #vector_db = FAISS.load_local("test_net_embedding", ollama_embeddings, allow_dangerous_deserialization=True)
    vectors = vector_db.index.reconstruct_n(0, vector_db.index.ntotal)

    #similarity_matrix = normalize(cosine_similarity(vectors))
    similarity_matrix = cosine_similarity(vectors)

    #Create a DataFrame for the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=authors_df.index, columns=authors_df.index)


    # Convert the similarity matrix to a distance matrix
    #distance_matrix = 1 - similarity_matrix
    distance_matrix = cosine_distances(vectors)

    # Apply DBSCAN
    dbscan = DBSCAN(metric='precomputed', eps=0.1, min_samples=2)
    dbscan.fit(distance_matrix)
    # Extract the cluster labels
    cluster_labels = dbscan.labels_
    # Add the cluster labels to the original DataFrame
    authors_df['Cluster'] = cluster_labels

    # Convert the similarity DataFrame to float64
    similarity_df = similarity_df.astype('float64') 

    # Define a color palette for clusters
    num_clusters = authors_df['Cluster'].nunique()
    palette = sns.color_palette("Spectral", num_clusters)
    cluster_colors = {cluster: f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}' 
                    for cluster, color in zip(authors_df['Cluster'].unique(), palette)}

    # Initialize the network
    net = Network(height="750px", width="100%", bgcolor="#dddddd", font_color="black", filter_menu=True, notebook=True, cdn_resources='remote')
    net.toggle_hide_edges_on_drag(False)
    net.barnes_hut()

    # Create a set of full names for first and last authors from paper_main_authors for quick lookup
    highlight_authors = set(paper_main_authors['first_author']).union(set(paper_main_authors['last_author']))

    # Add nodes with custom hover information
    for idx, row in authors_df.iterrows():
        node = idx
        full_name = f"{row['forename']} {row['lastname']}"
        title = f"{full_name}: {row['affiliation']}"
        color = cluster_colors[row['Cluster']]
        
        # Check if the full name is in the highlight_authors set
        if full_name in highlight_authors:
            size = 50  # Highlighted node size
        else:
            size = 35  # Default node size
            
        net.add_node(node, label=full_name, title=title, color=color, size=size)

    # Define a uniform color for all edges
    edge_color = '#a6a6a6'  # Grey color for all edges

    # Add edges with similarity values as hover information
    for i, source in enumerate(similarity_df.index):
        for j, target in enumerate(similarity_df.columns):
            if i < j:  # Ensure each pair is considered only once
                similarity = similarity_df.loc[source, target]
                if similarity >= 0.999:  # Only add edges with similarity of 1
                    source_name = f"{authors_df.loc[source, 'forename']} {authors_df.loc[source, 'lastname']}"
                    target_name = f"{authors_df.loc[target, 'forename']} {authors_df.loc[target, 'lastname']}"
                    edge_title = f"Edge from {source_name} to {target_name}. Similarity: {similarity:.2f}"
                    net.add_edge(source, target, title=edge_title, value=similarity, color=edge_color)
    return net
