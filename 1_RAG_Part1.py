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
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

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
# MAGIC ## Using PDF for RAG

# COMMAND ----------

path = "./Understanding_Climate_Change.pdf"

# COMMAND ----------

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = FAISS.from_documents(texts, embedding_model)

    return vectorstore


chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

# COMMAND ----------

## create Retreiver
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

# Test the retrieval
query = "What is climate change?"
similar_documents = chunks_query_retriever.get_relevant_documents(query)
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using CSV for RAG

# COMMAND ----------

import pandas as pd

file_path = ('./customers-100.csv')
data = pd.read_csv(file_path)

#preview the csv file
data.head()

# COMMAND ----------

loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()


## setup FAISS
index = faiss.IndexFlatL2(len(embedding_model.embed_query(" ")))
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)


## Add the splitted csv data to the vector store
vector_store.add_documents(documents=docs)

# COMMAND ----------

## Create retrival chain

retriever = vector_store.as_retriever()

# Set up system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    
])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

## Lets query

answer= rag_chain.invoke({"input": "which company does ponnama work for?"})
answer['answer']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check reliability of RAG ( retrieval correctness )

# COMMAND ----------

# Docs to index
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag",
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # number of documents to retrieve
            )

# COMMAND ----------

### Lets test 
question = "what are the differnt kind of agentic design patterns?"
docs = retriever.invoke(question)
docs

# COMMAND ----------

#### Check What our doc looks like

print(f"Title: {docs[0].metadata['title']}\n\nSource: {docs[0].metadata['source']}\n\nContent: {docs[0].page_content}\n")

# COMMAND ----------

## Check document relevancy
os.environ['GROQ_API_KEY'] = dbutils.secrets.get("sandesk4", "pat_token")

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatGroq(model="databricks-meta-llama-3-3-70b-instruct", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


#### We filter out non relevant docs here. 
docs_to_use = []
for doc in docs:
    print(doc.page_content, '\n', '-'*50)
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res,'\n')
    if res.binary_score == 'yes':
        docs_to_use.append(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC We follow the same ChatGroq module and similar PROMPT & function to address hallucinations 
# MAGIC Idea is to have ground truths ++ Retriver generated answers ++ then have LLM compare semnatic meanings and come up with YES or NO. If No we remove those documnets from the retrived list 

# COMMAND ----------


