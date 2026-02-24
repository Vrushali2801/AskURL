"""
Document processing utilities for loading and vectorizing content from URLs
"""
from typing import List, Optional
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from config import (
    GITHUB_TOKEN,
    BASE_URL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_INDEX_PATH
)


class DocumentProcessor:
    """Handles loading, splitting, and vectorizing documents from URLs"""
    
    def __init__(self):
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings"""
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=GITHUB_TOKEN,
                base_url=BASE_URL
            )
        return self.embeddings
    
    def load_urls(self, urls: List[str]) -> List:
        """
        Load content from a list of URLs
        
        Args:
            urls: List of URL strings to load
            
        Returns:
            List of loaded documents
        """
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        return documents
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, documents: List) -> FAISS:
        """
        Create a FAISS vector store from documents
        
        Args:
            documents: List of document chunks
            
        Returns:
            FAISS vector store
        """
        embeddings = self._initialize_embeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    
    def save_vectorstore(self, vectorstore: FAISS, file_path: Optional[str] = None):
        """
        Save FAISS vector store to disk using FAISS's native save method
        
        Args:
            vectorstore: FAISS vector store to save
            file_path: Path to save the index (defaults to config path, without extension)
        """
        if file_path is None:
            file_path = FAISS_INDEX_PATH.replace('.pkl', '')  # Remove .pkl extension
        
        # FAISS has its own save_local method
        vectorstore.save_local(file_path)
    
    def load_vectorstore(self, file_path: Optional[str] = None) -> Optional[FAISS]:
        """
        Load FAISS vector store from disk using FAISS's native load method
        
        Args:
            file_path: Path to the index folder (defaults to config path)
            
        Returns:
            FAISS vector store or None if file doesn't exist
        """
        if file_path is None:
            file_path = FAISS_INDEX_PATH.replace('.pkl', '')
        
        if self.index_exists(file_path):
            embeddings = self._initialize_embeddings()
            vectorstore = FAISS.load_local(
                file_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        return None
    
    def index_exists(self, file_path: Optional[str] = None) -> bool:
        """
        Check if a saved index exists
        
        Args:
            file_path: Path to check (defaults to config path)
            
        Returns:
            True if index file exists, False otherwise
        """
        if file_path is None:
            file_path = FAISS_INDEX_PATH.replace('.pkl', '')
        
        # FAISS creates a folder with index.faiss and index.pkl files
        return os.path.exists(file_path) and os.path.isdir(file_path)
    
    def process_urls(self, urls: List[str], save_index: bool = True) -> tuple:
        """
        Complete pipeline to process URLs into a vector store
        
        Args:
            urls: List of URLs to process
            save_index: Whether to save the index to disk
            
        Returns:
            Tuple of (vectorstore, num_documents, num_chunks)
        """
        # Load documents
        documents = self.load_urls(urls)
        num_documents = len(documents)
        
        # Split into chunks
        chunks = self.split_documents(documents)
        num_chunks = len(chunks)
        
        # Create vector store
        vectorstore = self.create_vectorstore(chunks)
        
        # Save if requested
        if save_index:
            self.save_vectorstore(vectorstore)
        
        return vectorstore, num_documents, num_chunks