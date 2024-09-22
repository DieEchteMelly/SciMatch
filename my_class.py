from ollamaworking import documentloader_fromprompt, rephrasing_input_to_fit_pubmedapiwrapper
from makingmap import parsedInfotoDF, create_pmid_authors_df, draw_map, search_for_paper_main_authors, create_final_df, create_final_paper_df
from makingnetwork import lets_embed

class SciMatch:
    def __init__(self):
        pass

    def get_research_topic(self, research_topic, top_k):
        docs = documentloader_fromprompt(research_topic, top_k)
        return docs
    
    def rephrase(self, research_topic):
        try:
            rephrased_query = rephrasing_input_to_fit_pubmedapiwrapper(research_topic)
            # Add debug print to verify the output
            print(f"Debug: Rephrased Query - {rephrased_query}")
            if rephrased_query is None:
                raise ValueError("Rephrasing returned None.")
            return rephrased_query
            
        except Exception as e:
            print(f"Error while rephrasing: {str(e)}")
            return None


    def document_to_df(self, docs):
        if docs is None:
            print("Debug: No documents to convert to DataFrame")
            raise ValueError("No documents found. Please check your input and try again.")
        parsedInfo = parsedInfotoDF(docs)
        return parsedInfo
    
    def get_paperInfo(self, parsedInfo):
        paper_df = create_final_paper_df(parsedInfo)
        return paper_df
    
    def get_authorsInfo_from_parsed_Info(self, parsedInfo):
        return create_pmid_authors_df(parsedInfo)
    
    def display_df(self, pmid_authors_df):
        displaying_df = create_final_df(pmid_authors_df)
        return displaying_df
    
    def authors_on_worldmap(self, pmid_authors_df):
        pmid_authors_df = pmid_authors_df.dropna(subset=['latitude', 'longitude'])
        return draw_map(pmid_authors_df)
    
    def define_main_authors(self, parsedInfo, pmid_authors_df):
        paper_main_authors, updated_pmid_authors_df = search_for_paper_main_authors(parsedInfo, pmid_authors_df)
        return paper_main_authors, updated_pmid_authors_df

    def authors_in_network(self, pmid_authors_df, parsed_info, paper_main_authors, ss_treshold):
        return lets_embed(pmid_authors_df, parsed_info, paper_main_authors, ss_treshold)
