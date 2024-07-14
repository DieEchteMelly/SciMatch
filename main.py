import streamlit as st
from my_class import SciMatch
from streamlit_folium import folium_static

header = st.container()
inputdata = st.container()
results = st.container()
visualization = st.container()

sci_match = SciMatch()

with header:
    st.title('Welcome to SciMatch!')
    st.text('In this project, we aim to find your perfect scientific matches!')
    st.write("Whether you're seeking collaboration or curious about your competition, just describe your topic to us, and we will find the best results for you.")
    st.write('SciMatch utilizes Ollama to search PubMed for publications based on your input. The results will identify scientists who have worked on these papers. You can view a basic list of these researchers, explore a world map to see where they are working globally, or analyze how closely related their work is.')
with inputdata:
    st.header('What are you looking for?')
    research_topic = st.text_input("Describe your research with an abstract, your title or throw in keywords:")
    top_k = st.slider("How many papers should we look for?", min_value=5, max_value=50, value=5, step=5)
    st.text('We are looking for your best matches. But good work takes a bit of time.')

    if research_topic:
        with st.spinner("Thinking ..."):
            try:
                top_k = int(top_k) if top_k else 5
                docs = sci_match.get_research_topic(research_topic, top_k)
                if docs is not None:
                    parsed_info = sci_match.document_to_df(docs)
                    pmid_authors_df = sci_match.get_authorsInfo_from_parsed_Info(parsed_info)
                    paper_main_authors, pmid_authors_df = sci_match.define_main_authors(parsed_info, pmid_authors_df)
                    st.success('Done!')
                else:
                    st.error("No documents found. Please check your input and try again.")
            except ValueError as e:
                st.error(str(e))

with results:
    st.header('These are the scientists out there with the most similar research to yours:')
    if 'pmid_authors_df' in locals():
        st.dataframe(pmid_authors_df)
        


with visualization:
    st.header('Here are your matches!')
    st.subheader('See where your matches are in the world!')
    st.write('You can explore the map to see where these scientists are located. By clicking on the location marker you can find out which scientists are located there.')
    if 'pmid_authors_df' in locals():
        world_map = sci_match.authors_on_worldmap(pmid_authors_df)
        folium_static(world_map) 
        

    st.subheader('Find the closest semantic similarity match here!')
    st.write('Here you can see the semantic similarity between the papers and thereby between the authors. Authors in one cluster are authors of one paper. The closer two clusters are, the more similar they are.')
    if 'pmid_authors_df' in locals():
        network = sci_match.authors_in_network(pmid_authors_df, parsed_info, paper_main_authors)
        network.show('graph.html')
        with open('graph.html', 'r') as f:
            html_string = f.read()
        st.components.v1.html(html_string, width=800, height=600)