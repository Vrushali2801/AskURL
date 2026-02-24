"""
Question-answering chain utilities for RAG-based querying
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from config import (
    GITHUB_TOKEN,
    BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVAL_K
)


class QAChain:
    """Handles question-answering using RAG with source attribution"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self._initialize_llm()
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}"""),
            ("human", "{question}")
        ])
        
        self.chain = self._create_chain()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=BASE_URL,
            temperature=LLM_TEMPERATURE
        )
        return llm
    
    def _format_docs(self, docs):
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _create_chain(self):
        """Create the RAG chain"""
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )
        return chain
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer
        
        Args:
            question: The question to ask
            
        Returns:
            Answer string
        """
        response = self.chain.invoke(question)
        return response.content
    
    def ask_with_sources(self, question: str) -> dict:
        """
        Ask a question and get answer with source attribution
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        # Use invoke() instead of get_relevant_documents()
        docs = self.retriever.invoke(question)
        
        # Get answer
        answer = self.ask(question)
        
        # Extract unique sources
        sources = []
        seen_sources = set()
        for doc in docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in seen_sources:
                    sources.append(source)
                    seen_sources.add(source)
        
        return {
            "answer": answer,
            "sources": "\n".join(sources) if sources else "No sources available"
        }
    
    def get_relevant_documents(self, question: str) -> list:
        """
        Get relevant documents for a question without generating answer
        
        Args:
            question: The question to search for
            
        Returns:
            List of relevant documents
        """
        # Use invoke() instead of get_relevant_documents()
        return self.retriever.invoke(question)