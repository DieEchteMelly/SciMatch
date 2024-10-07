import streamlit as st

st.image("scimatch_logo.png", width=400)
st.title("About SciMatch")

st.header("Overview")
st.markdown("SciMatch is a tool designed to bring scientists together. With LLM enhanced input optimization we search the PubMed database. We provide a refined list of publications and corresponding authors. In addition, scientists are displayed on a world map to provide an easy overview for people who are looking for collaboration partners but are limited to the countries of their collaboration partners. We also provide a network that shows how similar the research history of the retrieved authors is. ")
st.text("An overview of SciMatch's functionality is shown in the figure below:")
st.image("graphical_abstract.png",caption="Graphical Abstract")
st.markdown("""---""")
st.header("Code and Documentation")
st.page_link("https://github.com/DieEchteMelly/SciMatch", label="Visit our GitHub Repository")