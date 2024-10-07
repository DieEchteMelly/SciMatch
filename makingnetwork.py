import pandas as pd
from langchain.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
import requests, json
from langchain_community.vectorstores import FAISS
import numpy as np
import math
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

def lets_embed(pmid_authors_df, parsedInfo, paper_main_authors, ss_treshold):
    authors_df = pd.merge(pmid_authors_df, parsedInfo, left_on='pubmedid', right_on='uid', how='inner')
    authors_df['research'] = authors_df.apply(lambda row: str(row['Title'])+ ' ' + str(row['page_content']), axis=1)
    
    # Fuse authors by grouping them based on forename, lastname, and affiliation
    fused_authors_df = authors_df.groupby(['research']).agg({
        'forename': lambda x: ', '.join(x),  # Join all unique first names
        'lastname': lambda x: ', '.join(x),  # Join all unique last names
        'affiliation': lambda x: '; '.join(x),  # Join all unique last names
        'pubmedid': 'first',
        'research': 'count'    # Keep the first PubMed ID for reference
    }).rename(columns={'research': 'count'}).reset_index()
    # Convert all float32 columns to standard float (float64) in the dataframe
    fused_authors_df = fused_authors_df.applymap(lambda x: float(x) if isinstance(x, np.float32) else x)

    protocol = "https"
    hostname = "chat.cosy.bio"
    host = f"{protocol}://{hostname}"
    
    account = {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
    auth_url = f"{host}/api/v1/auths/signin"
    api_url = f"{host}/ollama"

    auth_response = requests.post(auth_url, json=account)
    jwt= json.loads(auth_response.text)["token"]

    ollama_embeddings = OllamaEmbeddings(base_url=api_url, model='nomic-embed-text', headers= {"Authorization": "Bearer " + jwt})

    loader = DataFrameLoader(fused_authors_df, page_content_column="research")
    vector_db = FAISS.from_documents(loader.load(), ollama_embeddings)
    vector_db.save_local("test_net_embedding")
    #vector_db = FAISS.load_local("test_net_embedding")
    vectors = vector_db.index.reconstruct_n(0, vector_db.index.ntotal)
    assert len(vectors) == len(fused_authors_df), "Mismatch between vector count and DataFrame length"
    #similarity_matrix = normalize(cosine_similarity(vectors))
    similarity_matrix = cosine_similarity(vectors)

    #Create a DataFrame for the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=fused_authors_df.index, columns=fused_authors_df.index)
    distance_matrix = cosine_distances(vectors)

    # Apply DBSCAN
    dbscan = DBSCAN(metric='precomputed', eps=0.1, min_samples=2)
    dbscan.fit(distance_matrix)
    # Extract the cluster labels
    fused_authors_df['Cluster']= dbscan.labels_
    # Define a color palette for clusters
    #num_clusters = fused_authors_df['Cluster'].nunique()
    def rgb_to_hex(rgb):
        """ Convert RGB tuple to HEX color code. """
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
    
    num_nodes = len(fused_authors_df)
    palette = sns.color_palette("crest", num_nodes)
    #cluster_colors = {cluster: f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}' 
    #                for cluster, color in zip(fused_authors_df['Cluster'].unique(), palette)}
    
    colors = [rgb_to_hex(color) for color in palette]
    # Initialize the network
    net = Network(height="750px", width="100%", bgcolor="#dddddd", font_color="black", filter_menu=True, notebook=True, cdn_resources='remote')
    net.toggle_physics(False)
    net.toggle_hide_edges_on_drag(False)

    # Create a set of full names for first and last authors from paper_main_authors for quick lookup
    first_author_dict = dict(zip(paper_main_authors['uid'], paper_main_authors['first_author']))

    # Add nodes with custom hover information
    prioritized_labels = {}
    for idx, row in fused_authors_df.iterrows():
        node_id = idx
        pubmed_id = row['pubmedid']  # Assuming this column exists in fused_authors_df
        forenames = row['forename'].split(",")
        lastnames = row['lastname'].split(",")
        affiliations = row.get('affiliation', 'No affiliation').split("; ")
        first_author = first_author_dict.get(pubmed_id, "Unknown Author")
        prioritized_labels[idx] = f"{first_author} et al."

        color = colors[idx]
        num_authors = min(len(forenames), len(lastnames), len(affiliations))
        
        # Get the first author for this paper
        first_author = first_author_dict.get(pubmed_id, "Unknown Author")

        # Create the node label
        node_label = f"{first_author} et al."

        # Construct the hover title with full names and affiliations
        labels = []
        for i in range(num_authors):
            forename = forenames[i].strip()
            lastname = lastnames[i].strip()
            affiliation = affiliations[i] if i < len(affiliations) else 'No affiliation'
            
            full_name = f"{forename} {lastname}"
            if full_name == first_author:
                labels.append(f"*{forename.upper()} {lastname.upper()}* (First Author): {affiliation}")
            else:
                labels.append(f"{forename.upper()} {lastname.upper()}: {affiliation}")

        # Join labels with newline characters
        hover_text = "\n".join(labels)
        
        # Set node size based on the count of research papers
        size = math.log(row['count']) * 10
            
        net.add_node(node_id, label=node_label, title=hover_text, color=color, size=size)

    # Define a uniform color for all edges
    edge_color = '#a6a6a6'  # Grey color for all edges

    # Add edges with similarity values as hover information
    for i, source in enumerate(similarity_df.index):
        for j, target in enumerate(similarity_df.columns):
            if i < j:  # Ensure each pair is considered only once
                similarity = float(similarity_df.loc[source, target])
                if similarity >= ss_treshold:  # Only add edges with similarity of 1
                    source_name = prioritized_labels[source]
                    target_name = prioritized_labels[target]
                    edge_title = f"Edge from '{source_name}' to '{target_name}'. Similarity: {similarity:.2f}"
                    net.add_edge(source, target, title=edge_title, value=0.1, color=edge_color)
    net.force_atlas_2based(gravity=0.01, central_gravity=0.001, spring_length=334, spring_strength=0.01, damping=0.01)
    return net