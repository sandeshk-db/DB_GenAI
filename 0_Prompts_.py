# Databricks notebook source
# MAGIC %md
# MAGIC From > https://github.com/NirDiamant/Prompt_Engineering

# COMMAND ----------

# MAGIC %pip install --upgrade langchain mlflow[databricks] langchain[docarray] databricks-sql-connector[sqlalchemy] langchain-community langchain-core langgraph docarray
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks
import os

from langchain.prompts import PromptTemplate

# Set the Databricks token from the secret scope
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("sandesk4", "pat_token")

# Initialize the embedding model with the specified endpoint
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# Initialize the chat model with the specified endpoint and max tokens
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 500)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prompt Basics

# COMMAND ----------

## Basic prompt
basic_prompt = "What is UC in Databricks?"
print(chat_model.invoke(basic_prompt).content)

# COMMAND ----------

## More structured prompt
structured_prompt = PromptTemplate(
    input_variables=["input"],
    template="Provide a definition of {input}, explain its importance, and list three key benefits."
)

chain = structured_prompt | chat_model # Combine the prompt template with the language model
input_variables = {"input": "UC in Databricks"} # Define the input variables
output = chain.invoke(input_variables).content # Invoke the chain with the input variables
print(output)

# COMMAND ----------

### factual check prompts

fact_check_prompt = PromptTemplate(
    input_variables=["statement"],
    template="""Evaluate the following statement for factual accuracy. If it's incorrect, provide the correct information:
    Statement: {statement}
    Evaluation:"""
)

chain = fact_check_prompt | chat_model
print(chain.invoke("The capital of France is Italy.").content)

# COMMAND ----------

## solving a complex problem

problem_solving_prompt = PromptTemplate(
    input_variables=["problem"],
    template="""Solve the following problem step by step:
    Problem: {problem}
    Solution:
    1)"""
)

chain = problem_solving_prompt | chat_model

print(chain.invoke("Calculate the compound interest on $1000 invested for 5 years at an annual rate of 5%, compounded annually.").content)
