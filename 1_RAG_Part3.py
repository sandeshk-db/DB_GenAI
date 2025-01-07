# Databricks notebook source
# MAGIC %pip install --upgrade langchain mlflow[databricks] langchain[docarray] databricks-sql-connector[sqlalchemy] langchain-community langchain-core langgraph docarray evaluation pypdf faiss-cpu chromadb langchain_groq
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks
import os
from langchain.prompts import PromptTemplate



from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

# Set the Databricks token from the secret scope
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("sandesk4", "pat_token")

# Initialize the embedding model with the specified endpoint
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# Initialize the chat model with the specified endpoint and max tokens
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 500)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improving Retrieval In RAG Systems

# COMMAND ----------

##1 - Query Rewriting: Reformulating queries to improve retrieval.

#re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Create a prompt template for query rewriting
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

# Create an LLMChain for query rewriting
query_rewriter = query_rewrite_prompt | chat_model

def rewrite_query(original_query):
    """
    Rewrite the original query to improve retrieval.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The rewritten query
    """
    response = query_rewriter.invoke(original_query)
    return response.content
  

# example query over the understanding climate change dataset
original_query = "What are the impacts of climate change on the environment?"
rewritten_query = rewrite_query(original_query)
print("Original query:", original_query)
print("\nRewritten query:", rewritten_query)

# COMMAND ----------

### 2 - Step-back Prompting: Generating broader queries for better context retrieval.

