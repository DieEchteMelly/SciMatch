import pandas as pd
import re
from Bio import Entrez
from pycountry import countries
from geopy.geocoders import Nominatim
from itables import show, init_notebook_mode
import numpy as np
import folium
from folium.plugins import MarkerCluster
import geojson
import json

pd.set_option('display.max_colwidth',None) 

def parsedInfotoDF(docs):
    data_dicts=[]
    for data_document in docs: #iterate through all entry in list "source document"
        dict_of_document = dict(data_document.dict(), **data_document.dict()["metadata"]) #make a dict out of the entries in the list and add metadata additionally so that we dont have a dict in a dict
        del dict_of_document["metadata"] #delete the dict in dict metadata
        del dict_of_document["type"] #delete type
        data_dicts.append(dict_of_document) #add this dict to the empty list
    data_dicts #shows list of dicts
    parsedInfo = pd.DataFrame.from_records(data_dicts) #create dataframe from the list of dicts
    parsedInfo.set_index('uid') # set uid as index
    return parsedInfo

def search_for_paper_main_authors(parsedInfo):

    Entrez.email = "melanie.altmann@studium.uni-hamburg.de"
    PubMedUIDs = parsedInfo['uid'].tolist() #create list of PubMed-UIDs from parsedInfo df
    handle = Entrez.efetch(db="pubmed", id=PubMedUIDs, retmode="xml")
    records = Entrez.read(handle)
    pmid_authors = {}

    for record in records["PubmedArticle"]:
        pmid = record["MedlineCitation"]["PMID"]
        authors = record["MedlineCitation"]["Article"]["AuthorList"]
        author_affiliations = []
        for author in authors:
            affiliations = [info['Affiliation'] for info in author['AffiliationInfo']]
            author_affiliations.append(f"{author['ForeName']} {author['LastName']}: {', '.join(affiliations)}")
        pmid_authors[pmid] = author_affiliations
    #the following part is neccasary for highlighting the foírst and the last author in my network
    main_authors = []
    for uid, authors in pmid_authors_df.iterrows():
        first_author = None
        last_author = None

    # Get the first author
    for author in authors:
        if author:  # Check if the author string is not None or empty
            first_author = author
            break

    # Get the last author
    for author in reversed(authors):
        if author:  # Check if the author string is not None or empty
            last_author = author
            break

    main_authors.append({'uid': uid, 'first_author': first_author, 'last_author': last_author})

    # Create DataFrame from the main_authors list
    paper_main_authors = pd.DataFrame(main_authors)

    # Clean the first_author and last_author fields
    paper_main_authors['first_author'] = paper_main_authors['first_author'].str.split(':').str[0].str.strip()
    paper_main_authors['last_author'] = paper_main_authors['last_author'].str.split(':').str[0].str.strip()
    paper_main_authors.to_pickle("./paper_main_authors.pkl") 
    return paper_main_authors

def create_pmid_authors_df(parsedInfo):
    Entrez.email = "melanie.altmann@studium.uni-hamburg.de"
    PubMedUIDs = parsedInfo['uid'].tolist()  # create list of PubMed-UIDs from parsedInfo df
    handle = Entrez.efetch(db="pubmed", id=PubMedUIDs, retmode="xml")
    records = Entrez.read(handle)

    author_data = []

    for record in records["PubmedArticle"]:
        pmid = record["MedlineCitation"]["PMID"]
        authors = record["MedlineCitation"]["Article"]["AuthorList"]
        for author in authors:
            affiliations = [info['Affiliation'] for info in author['AffiliationInfo']]
            for affiliation in affiliations:
                author_data.append({
                    'forename': author['ForeName'],
                    'lastname': author['LastName'],
                    'affiliation': affiliation,
                    'pubmedid': pmid
                })

    pmid_authors_df = pd.DataFrame(author_data)
    pmid_authors_df = pmid_authors_df.sort_values(by='lastname')
    def extract_country(affiliation):
        try:
            country = affiliation.split(',')[-1].strip()
            country = country[:-1]
            return country
        except:
            return None
        
    def extract_city(affiliation):
        try:
            city = affiliation.split(',')[-2].strip()
            return city
        except:
            return None

    # Using vectorized operations
    pmid_authors_df['country'] = pmid_authors_df['affiliation'].apply(extract_country)
    pmid_authors_df['city'] = pmid_authors_df['affiliation'].apply(extract_city)

    # Using list comprehension
    pmid_authors_df['country'] = [extract_country(aff) for aff in pmid_authors_df['affiliation']]
    pmid_authors_df['city'] = [extract_city(aff) for aff in pmid_authors_df['affiliation']]
    
    def replace_country_codes(df):
        # Create a dictionary mapping country codes to country names
        country_dict = {country.alpha_3: country.name for country in countries}

        # Replace country codes with names in the 'country' column
        df['country'] = df['country'].apply(lambda code: country_dict.get(code.upper(), code))
    replace_country_codes(pmid_authors_df)
    geolocator = Nominatim(user_agent="my_application")
    def get_coordinates(row):
        try:
            # Concatenate city and country
            location = geolocator.geocode(f"{row['city']}, {row['country']}")
            if location:
                return pd.Series([location.latitude, location.longitude])
            else:
                return pd.Series([None, None])
        except:
            return pd.Series([None, None])

    # Apply the function to the dataframe
    pmid_authors_df[['latitude', 'longitude']] = pmid_authors_df.apply(get_coordinates, axis=1)
    def merge_and_deduplicate(df):
        # Group by the specified columns
        grouped = pmid_authors_df.groupby(['forename', 'lastname', 'pubmedid'])
        
        # Define a function to merge affiliations
        def merge_affiliations(group):
            merged_affiliation = ' '.join(group['affiliation'].dropna().unique())
            first_row = group.iloc[0]
            first_row['affiliation'] = merged_affiliation
            return first_row
        
        # Apply the merging function and reset the index
        result = grouped.apply(merge_affiliations).reset_index(drop=True)
        
        return result
    pmid_authors_df = merge_and_deduplicate(pmid_authors_df)
    return pmid_authors_df

def draw_map(pmid_authors_df):
    pmid_authors_deduplicated_df = pmid_authors_df.copy()
    grouped = pmid_authors_deduplicated_df.groupby(['forename', 'lastname'])
    pmid_authors_deduplicated_df = grouped.first().reset_index()
    pmid_authors_deduplicated_df
    n = folium.Map(location=(30,10), zoom_start=2.3, tiles="cartodb positron")
    with open('ne_50m_admin_0_countries.geojson') as f:
        world = json.load(f)

    # Create a dictionary to store affiliations for each location
    location_affiliations = {}

    for i in range(len(pmid_authors_df)):
        lat = pmid_authors_df.iloc[i]['latitude']
        lon = pmid_authors_df.iloc[i]['longitude']
        affiliation = pmid_authors_df.iloc[i]['affiliation']
        forename = pmid_authors_df.iloc[i]['forename']
        lastname = pmid_authors_df.iloc[i]['lastname']
        location = (lat, lon)
    

        # Create the popup text with affiliation, forename, and lastname
        popup_text = f"<b>{forename} {lastname}</b>: {affiliation}"

        # Add the popup text to the list for the current location
        if location in location_affiliations:
            location_affiliations[location].append(popup_text)
        else:
            location_affiliations[location] = [popup_text]

    # Add markers with multiple affiliations
    for location, popup_texts in location_affiliations.items():
        lat, lon = location
        popup_text = '<br>'.join(popup_texts)
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(n)
    return n