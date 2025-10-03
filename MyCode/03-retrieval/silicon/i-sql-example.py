"""
The below example will use a SQLite connection with the Chinook database, which is a sample database that represents a digital media store. Follow these installation steps to create Chinook.db in the same directory as this notebook. You can also download and build the database via the command line:

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

Afterwards, place `Chinook.db` in the same directory where this code is running.

"""

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
# replace this with the connection details of your db
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import re
load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")
embedding_model_name = os.environ.get("EMBEDDING_MODEL")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.get_usable_table_names())
llm = ChatOpenAI(model=model_name,base_url=base_url,api_key=api_key, temperature=0)

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)

# combined chain = write_query | execute_query
combined_chain = write_query | execute_query

# run the chain
result = combined_chain.invoke({"question": "How many employees are there?"})

print(result)
