import streamlit as st
from my_class import SciMatch
from streamlit_folium import folium_static
import os
import pandas as pd

header = st.container()
inputdata = st.container()
results = st.container()
visualization = st.container()

sci_match = SciMatch()

if "rephrased_topic" not in st.session_state:
    st.session_state.rephrased_topic = ""

with header:
    st.image("scimatch_logo.png", width=400)
    st.title('Welcome to SciMatch!')
    st.write('In this project, we aim to find your perfect scientific matches!')
    st.write("Whether you're seeking collaboration or curious about your competition, just describe your topic to us, and we will find the best results for you.")
    st.write('SciMatch searches in PubMed for publications based on your input. The results will identify scientists who have worked on these papers. You can view a basic list of these researchers, explore a world map to see where they are working globally, or analyze how closely related their work is.')

with inputdata:
    st.header('What are you looking for?')
    top_k = st.slider("How many papers should we look for? Keep in mind, that the processing time increases with the number of papers to search for.", min_value=3, max_value=30, value=3, step=1)
    st.write("The quality of your results will depend on the quality of your topic description. It is helpful to keep your input simple. You might want to consider using the following scheme if you have a very specific topic: {X} AND ({Y} OR {Z}). If you're still getting no results, you could try using the 'Rephrase with AI' button.")
    research_topic = st.text_input("Your research topic goes here. Press Enter to confirm your search.", value=st.session_state.rephrased_topic)
    if st.button('Rephrase with AI'):
        with st.spinner("Rephrasing..."):
            try:
                rephrased_topic = sci_match.rephrase(research_topic)
                st.session_state.rephrased_topic = rephrased_topic
                st.success(f"Rephrasing complete!\n\nRephrased Query: {rephrased_topic}\n\nCopy this in the input field and press enter to confirm the new search. ")
            except Exception as e:
                st.error(f"Error during rephrasing: {e}")
        
    st.write('We are looking for your best matches. But good work takes a bit of time.')

    
    if research_topic:
        with st.spinner("Thinking ..."):
            try:
                top_k = int(top_k) if top_k else 5
                docs = sci_match.get_research_topic(research_topic, top_k)
                if docs is not None:
                    parsed_info = sci_match.document_to_df(docs)
                    pmid_authors_df = sci_match.get_authorsInfo_from_parsed_Info(parsed_info)
                    paper_main_authors, pmid_authors_df = sci_match.define_main_authors(parsed_info, pmid_authors_df)
                    displaying_df = sci_match.display_df(pmid_authors_df)
                else:
                    st.error("No documents found. Please check your input and try again.")
            except ValueError as e:
                st.error(str(e))

        with results:
            st.subheader(f"These are the results for your input '{research_topic}'.")
            st.subheader('These are the scientists out there with the most similar research to yours:')
            with st.expander('Watch your table of authors.'):
                if 'displaying_df' in locals():
                    st.write("You want to do a deep dive about an author? By clicking on the link you will come straight to the paper and there you can find the author.")
                    st.markdown(displaying_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
        with visualization:
            st.subheader('See where your matches are in the world!')
            st.write('You can explore the map to see where these scientists are located. By clicking on the location marker you can find out which scientists are located there.')
            with st.expander('Have a look at your worldmap.'):
                if 'pmid_authors_df' in locals():
                    world_map = sci_match.authors_on_worldmap(pmid_authors_df)
                    folium_static(world_map) 

            st.subheader('Find the closest semantic similarity match here!')
            st.write('Here you can see the semantic similarity between the papers and thereby between the authors. Authors in one cluster are authors of one paper. The highlighted authors in one clusters are the first and last author of the paper. The closer two clusters are, the more similar they are. You might need to zoom out at the beginning in order to see all clusters.')
            with st.expander('Dive into your network.'):
                if 'pmid_authors_df' in locals():
                    ss_treshold= st.slider("Set the treshold from which on you want to see edges between the similar authors.", min_value=0.1, max_value=1.0, value=0.4, step=0.01)
                    network = sci_match.authors_in_network(pmid_authors_df, parsed_info, paper_main_authors, ss_treshold)

                    if network:
                        network.show('graph.html')
                    else:
                        print("Error: Network is None")
                    with open('graph.html', 'r') as f:
                        html_string = f.read()
                    st.components.v1.html(html_string, width=800, height=600)
            st.link_button("Give us feedback!", "https://forms.gle/a2egtfM5MqoANRMCA")
