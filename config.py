"""
Configuration settings for the News Search Tool
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://models.inference.ai.azure.com"

# Model Configuration
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE = 0

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
RETRIEVAL_K = 3  # Number of documents to retrieve

# Persistence Configuration
FAISS_INDEX_PATH = "faiss_store_openai.pkl"

# Streamlit Configuration
PAGE_TITLE = "AskURL"
PAGE_ICON = "🔗"

# Prompt Template
QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question at the end.
If you don't know the answer or the context doesn't contain the information, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
