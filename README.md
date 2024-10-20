# SciMatch
<p align="center">
  <img src="https://github.com/DieEchteMelly/SciMatch/blob/main/scimatch_logo.png" width="300"/>
</p>

## Introduction
SciMatch is a tool designed to bring scientists together. With LLM enhanced input optimization we search the PubMed database. We provide a refined list of publications and corresponding authors. In addition, scientists are displayed on a world map to provide an easy overview for people who are looking for collaboration partners but are limited to the countries of their collaboration partners. We also provide a network that shows how similar the research history of the retrieved authors is.

<p align="center">
  <img src="https://github.com/DieEchteMelly/SciMatch/blob/main/graphical_abstract.png" width="300"/>
</p>

## Features

- LLM-enhanced input rephrasing
- Searches PubMed for best publications based on input
- Refined Lists of Publications and their authors
- World map with markers for the locations of each author
- Similarity network with each node representing authors with the same research history based on retrieved publications

## Requirements

- Python 3.x
- Additional Python libraries specified in `requirements.txt`
- Login Information to the Server (request those here: https://chat.cosy.bio/); need to be stored accordingly in the ".env" file

## Installation from GitHub

```bash
git clone https://github.com/DieEchteMelly/SciMatch.git
cd SciMatch
pip install -r requirements.txt
```

## Methods

SciMatch has the following functions:

- `get_research_topic()`
- `rephrase()`
- `document_to_df()`
- `get_paperInfo()`
- `get_authorsInfo_from_parsed_Info()`
- `display_df()`
- `authors_on_worldmap()`
- `define_main_authors()`
- `authors_in_network()`

## Contributing

If you find any bugs or wish to propose new features, please let us know.