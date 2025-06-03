import pandas as pd
from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# configuration
CHROMA_PATH = "chroma_db_langchain"
COLLECTION_NAME = "employee_profiles_langchain"
EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral" # Ensure this model is pulled in Ollama

class HRRAGSystem:
    """
    Encapsulates the LangChain RAG system for HR assistant functionalities.
    """
    def __init__(self):
        self.embedding_model = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.rag_chain = None
        self._initialize_components()

    def _initialize_components(self):
        """Initializes embedding model, ChromaDB, LLM, and the RAG chain."""
        print("Initializing HR RAG System components...")

        # initialize HuggingFace embeddings
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_HF)
        print(f"Embedding model loaded: {EMBEDDING_MODEL_HF}")

        # initialize ChromaDB client and collection
        try:
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_model,
                persist_directory=CHROMA_PATH
            )
            if self.vectorstore._collection.count() == 0:
                print("Warning: ChromaDB is empty. Please run the data ingestion script first to populate it.")
            else:
                print(f"ChromaDB initialized with {self.vectorstore._collection.count()} documents.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            print("Ensure 'chroma_db_langchain' directory exists and contains valid data.")
            self.vectorstore = None # Set to None if initialization fails

        # Initialize LLM with Ollama
        try:
            self.llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
            # Test Ollama connection
            test_response = self.llm.invoke("Hello")
            print(f"Ollama LLM ({OLLAMA_MODEL}) initialized. Test response: {test_response[:30]}...")
        except Exception as e:
            print(f"Error initializing Ollama LLM: {e}")
            print("Please ensure Ollama is installed and the model is pulled (`ollama pull {OLLAMA_MODEL}`).")
            self.llm = None

        # Define the prompt template for the LLM
        prompt_template = PromptTemplate.from_template(
            (
                "You are an intelligent HR assistant. "
                "Based on the following employee information, answer the user's query comprehensively. "
                "If the information is not sufficient to answer, state that you don't have enough data. "
                "Always try to provide names of relevant employees and their key attributes.\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )

        # Create a retriever from the vector store
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr", # "similarity" or "mmr" (Maximal Marginal Relevance)
                search_kwargs={"k": 5} # Retrieve top 5 most relevant documents
            )
            print("Retriever initialized.")
        else:
            print("Retriever not initialized due to missing vectorstore.")

        # Construct the RAG chain using LangChain Expression Language (LCEL)
        if self.retriever and self.llm:
            self.rag_chain = (
                {"context": self.retriever | RunnableLambda(self._format_docs), "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            print("LangChain RAG chain constructed.")
        else:
            print("RAG chain could not be constructed due to missing retriever or LLM.")

    def _format_docs(self, docs: List[Document]) -> str:
        """Formats retrieved documents for the LLM context."""
        formatted_str = ""
        if not docs:
            return "No relevant employee information found."
        for i, doc in enumerate(docs):
            formatted_str += f"--- Employee {i+1} ---\n"
            # Access metadata directly from doc.metadata
            formatted_str += f"Name: {doc.metadata.get('name', 'N/A')}\n"
            formatted_str += f"Skills: {doc.metadata.get('skills', 'N/A')}\n"
            formatted_str += f"Experience: {doc.metadata.get('experience_years', 'N/A')} years\n"
            formatted_str += f"Past Projects: {doc.metadata.get('past_projects', 'N/A')}\n"
            formatted_str += f"Availability: {doc.metadata.get('availability', 'N/A')}\n"
            formatted_str += "\n"
        return formatted_str.strip()

    async def query_chatbot(self, user_query: str) -> str:
        """Invokes the RAG chain to get a response from the chatbot."""
        if not self.rag_chain:
            raise RuntimeError("RAG chain is not initialized. Cannot process query.")
        return await self.rag_chain.ainvoke(user_query)

    async def search_employees_semantic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search for employees using the vector store.
        Returns a list of dictionaries representing employee data.
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store is not initialized. Cannot search employees.")
        
        # Use asimilarity_search for async operation
        retrieved_docs = await self.vectorstore.asimilarity_search(query, k=top_k)

        found_employees_data = []
        for doc in retrieved_docs:
            employee_data = doc.metadata
            # Ensure consistency with how it's expected by the Employee Pydantic model
            found_employees_data.append({
                "name": employee_data.get('name', 'N/A'),
                "skills": employee_data.get('skills', 'N/A'),
                "experience_years": employee_data.get('experience_years', 0),
                "past_projects": employee_data.get('past_projects', 'N/A'),
                "availability": employee_data.get('availability', 'N/A')
            })
        return found_employees_data
