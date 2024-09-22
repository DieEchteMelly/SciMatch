import streamlit as st
from my_class import SciMatch
from streamlit_folium import folium_static
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

st.set_page_config(page_title="SciMatch", page_icon="🤝", layout="wide",initial_sidebar_state="collapsed")

mitte, rechts = st.columns(2)
sci_match = SciMatch()

if "rephrased_topic" not in st.session_state:
    st.session_state.rephrased_topic = ""

with mitte:
    with st.container():
        st.image("scimatch_logo.png", width=400)
        st.write("**Discover your scientific matches with SciMatch.** Describe your topic, and we'll identify researchers, map their locations, and analyze their semantic similarity.")
        with st.expander('See here for further explanation.', icon="ℹ️"):
            st.write("In this project, we aim to find your perfect scientific matches! Whether you're seeking collaboration or curious about your competition, just describe your topic to us, and we will find the best results for you. SciMatch searches in PubMed for publications based on your input. The results will identify scientists who have worked on these papers. You can view a basic list of these researchers, explore a world map to see where they are working globally, or analyze how closely related their work is.")
    with st.container():
        st.header('What are you looking for?')
        research_topic = st.text_input("Your research topic goes here. Press Enter to confirm your search.", value=st.session_state.rephrased_topic)
        with st.expander('Further explanation and declaration', icon="ℹ️"):
            top_k = st.slider("How many papers should we look for? Keep in mind, that the processing time increases with the number of papers to search for.", min_value=3, max_value=30, value=3, step=1)
            st.write("The quality of your results will depend on the quality of your topic description. It is helpful to keep your input simple. You might want to consider using the following scheme if you have a very specific topic: {X} AND ({Y} OR {Z}). If you're still getting no results, you could try using the 'Rephrase with AI' button.")
        
        if st.button('Rephrase with AI'):
            with st.spinner("Rephrasing..."):
                try:
                    rephrased_topic = sci_match.rephrase(research_topic)
                    st.session_state.rephrased_topic = rephrased_topic
                    st.success(f"Rephrasing complete!\n\nRephrased Query: {rephrased_topic}\n\nCopy this in the input field and press enter to confirm the new search. ")
                except Exception as e:
                    st.error(f"Error during rephrasing: {e}")
        # Add the "Run SciMatch" button below the "Rephrase with AI" button
        if st.button('Run SciMatch',type="primary"):
            if research_topic:
                progress_text = "Initializing"
                progress_bar = st.progress(0, text=progress_text)  # Initialize the progress bar at 0%

                try: 
                    top_k = int(top_k) if top_k else 5
                    progress_text = "Setting up search parameters 🔍"
                    progress_bar.progress(10, text=progress_text)  
                    
                    progress_text = "Fetching documents from the database 📨"
                    docs = sci_match.get_research_topic(research_topic, top_k)
                    progress_bar.progress(30, text=progress_text) 
                    
                    if docs is not None:
                        progress_text = "Processing documents "
                        parsed_info = sci_match.document_to_df(docs)
                        progress_bar.progress(50, text=progress_text)  
                        
                        progress_text = "Retrieving paper information 📄"
                        paper_df = sci_match.get_paperInfo(parsed_info)
                        progress_bar.progress(70, text=progress_text) 
                        
                        progress_text = "Retrieving author information 🧑‍🔬"
                        pmid_authors_df = sci_match.get_authorsInfo_from_parsed_Info(parsed_info)
                        progress_bar.progress(90, text=progress_text)  

                        paper_main_authors, pmid_authors_df = sci_match.define_main_authors(parsed_info, pmid_authors_df)
                        
                        progress_text = "Finalizing the results 🏁"
                        displaying_df = sci_match.display_df(pmid_authors_df)
                        progress_bar.progress(100, text=progress_text)  
                    else:
                        st.error("No documents found. Please check your input and try again.")
                        progress_bar.empty()  # Remove the progress bar if no documents are found

                except ValueError as e:
                    st.error(str(e))
                    progress_bar.empty()  # Remove the progress bar if there's an error

            # Display the retrieved results and papers in tables as in your original code
            with st.container():
                st.subheader(f"These are the results for your input '{research_topic}'.")
                
                st.subheader('These are the papers we retrieved based on your input:')
                with st.expander('Table of retrieved papers'):
                    if 'paper_df' in locals():
                        gb = GridOptionsBuilder.from_dataframe(paper_df)
                        gb.configure_default_column(filter='agMultiColumnFilter', editable=True)
                        gb.configure_column('Title', width=300, maxWidth=300, resizable=True, tooltipField='Title', cellStyle={'white-space': 'nowrap', 'text-overflow': 'ellipsis', 'overflow': 'hidden'})
                        gb.configure_column('Abstract', width=300, maxWidth=300, resizable=True, tooltipField='Abstract', cellStyle={'white-space': 'nowrap', 'text-overflow': 'ellipsis', 'overflow': 'hidden'})
                        
                        gridOptions = gb.build()
                        AgGrid(
                            paper_df,
                            gridOptions=gridOptions,
                            update_mode=GridUpdateMode.MODEL_CHANGED,
                            data_return_mode=DataReturnMode.FILTERED,
                            enable_enterprise_modules=True,  
                            theme='alpine',
                            height=400,
                            reload_data=True
                            )
                
                st.subheader('These are the scientists out there with the most similar research to yours:')
                with st.expander('Table of retrieved authors'):
                    if 'displaying_df' in locals():
                        gb = GridOptionsBuilder.from_dataframe(displaying_df)
                        gb.configure_column('Country', filter='agSetColumnFilter', filter_params={"buttons": ["clear", "apply"]})
                        gb.configure_default_column(filter='agMultiColumnFilter', editable=True)
                        gridOptions = gb.build()
                        AgGrid(
                            displaying_df,
                            gridOptions=gridOptions,
                            update_mode=GridUpdateMode.MODEL_CHANGED,
                            data_return_mode=DataReturnMode.FILTERED,
                            enable_enterprise_modules=True,
                            theme='alpine',
                            height=400,
                            reload_data=True
                        )

with rechts:
            
    with st.container():
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