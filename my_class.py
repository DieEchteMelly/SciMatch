from ollamaworking import documentloader_fromprompt
from makingmap import parsedInfotoDF, create_pmid_authors_df, draw_map, search_for_paper_main_authors
from makingnetwork import lets_embed


class SciMatch:
    def __init__(self):
        pass

    def get_research_topic(self, research_topic, top_k):
        print(f"Debug: Calling documentloader_fromprompt with research_topic='{research_topic}' and top_k='{top_k}'")
        docs = documentloader_fromprompt(research_topic, top_k)
        if docs is None:
            print("Debug: documentloader_fromprompt returned None")
        else:
            print(f"Debug: documentloader_fromprompt returned {len(docs)} documents")
        return docs
    
    def document_to_df(self, docs):
        if docs is None:
            print("Debug: No documents to convert to DataFrame")
            raise ValueError("No documents found. Please check the input and try again.")
        parsedInfo = parsedInfotoDF(docs)
        return parsedInfo
    
    def get_authorsInfo_from_parsed_Info(self, parsedInfo):
        return create_pmid_authors_df(parsedInfo)
    
    def authors_on_worldmap(self, pmid_authors_df):
        pmid_authors_df = pmid_authors_df.dropna(subset=['latitude', 'longitude'])
        return draw_map(pmid_authors_df)
    
    def define_main_authors(self, parsedInfo, pmid_authors_df):
        paper_main_authors, updated_pmid_authors_df = search_for_paper_main_authors(parsedInfo, pmid_authors_df)
        print(f"Debug: define_main_authors() paper_main_authors columns: {paper_main_authors.columns}")
        return paper_main_authors, updated_pmid_authors_df

    def authors_in_network(self, pmid_authors_df, parsedInfo, paper_main_authors):
        print(f"Debug: authors_in_network() paper_main_authors type: {type(paper_main_authors)}")
        print(f"Debug: authors_in_network() paper_main_authors columns: {paper_main_authors.columns}")
        return lets_embed(pmid_authors_df, parsedInfo, paper_main_authors)