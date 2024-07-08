#!/usr/bin/env python3
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
print(llm.invoke("how can langsmith help with testing?"))