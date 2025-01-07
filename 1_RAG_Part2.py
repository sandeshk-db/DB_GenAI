# Databricks notebook source
# MAGIC %pip install --upgrade langchain mlflow[databricks] langchain[docarray] databricks-sql-connector[sqlalchemy] langchain-community langchain-core langgraph docarray evaluation pypdf faiss-cpu chromadb langchain_groq
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import faiss
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq


import random
import string
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sklearn.metrics import precision_score, recall_score

### Build Index
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings



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
# MAGIC ## Checking if chunk size affects the accuracies of retieval !

# COMMAND ----------

# Create a random dataset (e.g., 1000 documents with 10 sentences each)
def generate_dataset(num_documents=1000, num_sentences=10, sentence_length=10):
    dataset = []
    for _ in range(num_documents):
        doc = []
        for _ in range(num_sentences):
            sentence = ''.join(random.choices(string.ascii_lowercase + " ", k=sentence_length))
            doc.append(sentence)
        dataset.append(doc)
    return dataset

# Chunking the dataset into different chunk sizes (by sentences)
def chunk_dataset(dataset, chunk_size):
    chunks = []
    for doc in dataset:
        for i in range(0, len(doc), chunk_size):
            chunk = " ".join(doc[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

# Embed the dataset using OpenAI embeddings
def embed_dataset(chunks, embedding_model):
    embeddings = embedding_model.embed_documents(chunks)
    return np.array(embeddings)

# Set up FAISS index
def setup_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Using L2 distance (Euclidean distance)
    index.add(embeddings)  # Adding embeddings to FAISS index
    return index

# Perform a query and get the closest documents
def query_faiss_index(query, index, embedding_model, top_k=5):
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array([query_embedding])
    _, indices = index.search(query_embedding, top_k)
    return indices

# Evaluate performance metrics
def evaluate_retrieval(dataset, chunk_size, index, embedding_model, top_k=5):
    # Generate a random query
    query = " ".join(random.choice(dataset)[random.randint(0, len(dataset[0])-1)].split())
    
    # Retrieve top-k results using FAISS index
    indices = query_faiss_index(query, index, embedding_model, top_k)
    
    # Create ground truth by finding relevant chunks (naive evaluation for simplicity)
    ground_truth = [i for i in range(len(dataset)) if query in " ".join(dataset[i])]
    
    # Simulate precision and recall (based on simple matching)
    y_true = [1 if i in ground_truth else 0 for i in indices[0]]
    y_pred = [1] * len(indices[0])
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return precision, recall

# Main Experiment Function
def run_experiment(chunk_sizes=[1, 5, 10, 20]):
    # Create a random dataset
    dataset = generate_dataset(num_documents=1000, num_sentences=10, sentence_length=10)
    
    # Set up OpenAI Embedding model
    #embedding_model = OpenAIEmbeddings(openai_api_key="your_openai_api_key")

    for chunk_size in chunk_sizes:
        print(f"Running experiment with chunk size {chunk_size}...")
        
        # Chunk the dataset
        chunks = chunk_dataset(dataset, chunk_size)
        
        # Embed the chunks
        embeddings = embed_dataset(chunks, embedding_model)
        
        # Set up FAISS index
        index = setup_faiss_index(embeddings)
        
        # Evaluate retrieval performance
        precision, recall = evaluate_retrieval(dataset, chunk_size, index, embedding_model)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

# Run experiment with different chunk sizes
run_experiment(chunk_sizes=[1, 5, 10, 20])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Propositions Chunking -- Using LLM for chunking. 
# MAGIC
# MAGIC The system break downs the input text into propositions that are atomic, factual, self-contained, and concise in nature, encodes the propositions into a vectorstore, which can be later used for retrieval
# MAGIC

# COMMAND ----------

import re
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Text chunking function: Split sentences and chunk propositions based on logical connectors
def chunk_into_propositions(text):
    # Use regex to split sentences by punctuation marks (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # We will split each sentence further by logical connectors: 'and', 'but', 'so', etc.
    propositions = []
    for sentence in sentences:
        # Split by common conjunctions to get propositions
        sub_chunks = re.split(r'\b(and|but|so|because|although|however)\b', sentence)
        propositions.extend([chunk.strip() for chunk in sub_chunks if chunk.strip()])
    
    return propositions

# Example text
text = """
The dog barked loudly at the mailman. It was a sunny afternoon, and the children were playing outside. 
Suddenly, the dog stopped barking and ran inside because it saw a squirrel.
"""

# 1. Chunk the text into propositions
propositions = chunk_into_propositions(text)
print("Propositions:", propositions)

# 2. Set up Langchain embeddings ( We use databricks foundational models )
#embedding_model = OpenAIEmbeddings(openai_api_key="your_openai_api_key")

# 3. Embed the propositions
proposition_embeddings = embedding_model.embed_documents(propositions)

# Create tuples of (text, embedding)
text_embedding_tuples = list(zip(propositions, proposition_embeddings))

# 4. Set up FAISS index to store and retrieve propositions
faiss_index = FAISS.from_embeddings(text_embedding_tuples, embedding_model)

# 5. Example: Retrieve the most relevant proposition for a query
query = "What did the dog do after barking?"
query_embedding = embedding_model.embed_query(query)

# Convert query_embedding to a NumPy array
query_embedding = np.array(query_embedding).reshape(1, -1)

# Retrieve top 3 most relevant propositions
top_k = 3
_, indices = faiss_index.index.search(query_embedding, top_k)

# Print out the top k relevant propositions
print("Top relevant propositions:")
for idx in indices[0]:
    print(propositions[idx])

# COMMAND ----------

# MAGIC %md
# MAGIC Proposition chunking is a method of splitting long sentences. 
# MAGIC
# MAGIC Ex : Sandesh went to manali and visited Solan valley
# MAGIC  === > Broken down into :
# MAGIC -  Sandesh went to manali
# MAGIC - Sandesh visited solng valley
# MAGIC
# MAGIC Basically , the sentences are broken down into meaningful complete sentences rather than refferals
# MAGIC
# MAGIC - Later we can run the evaluation retreiver and check if this works better. 
