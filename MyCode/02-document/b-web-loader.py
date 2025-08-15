"""
Install the beautifulsoup4 package:

pip install beautifulsoup4
"""

import os
from langchain_community.document_loaders import WebBaseLoader

os.environ["USER_AGENT"] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'

loader = WebBaseLoader('https://www.langchain.com/')
docs = loader.load()

print(docs)
