import pandas as pd
from typing import List, Dict, Any

# Import necessary components from LangChain libraries
from langchain_huggingface import HuggingFaceEmbeddings # For creating embeddings from text using HuggingFace models
from langchain_community.llms import Ollama # For interacting with Ollama-hosted Large Language Models
from langchain_chroma import Chroma # For using ChromaDB as a vector store
from langchain_core.documents import Document # Base class for documents in LangChain
from langchain_core.prompts import PromptTemplate # For defining structured prompts for LLMs
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # For building LangChain Expression Language (LCEL) chains
from langchain_core.output_parsers import StrOutputParser # For parsing string output from LLMs

# --- Configuration Constants ---
# Define constants for file paths, collection names, and model names for easy modification.
CHROMA_PATH = "chroma_db_langchain" # Directory where ChromaDB will persist its data
COLLECTION_NAME = "employee_profiles_langchain" # Name of the collection within ChromaDB
EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2" # HuggingFace model for generating embeddings
OLLAMA_MODEL = "mistral" # Name of the LLM model to use from Ollama (e.g., "mistral", "llama2")

class HRRAGSystem:
    """
    Encapsulates the entire LangChain-based Retrieval Augmented Generation (RAG) system
    for HR assistant functionalities. This class manages the embedding model,
    vector store (ChromaDB), Large Language Model (LLM), and the RAG chain itself.
    """
    def __init__(self):
        # Initialize instance variables to None; they will be populated during initialization.
        self.embedding_model = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.rag_chain = None
        # Call the private initialization method to set up all components.
        self._initialize_components()

    def _initialize_components(self):
        """
        Initializes the embedding model, ChromaDB (vector store), Ollama LLM,
        and constructs the RAG chain. This method handles potential errors
        during component loading.
        """
        print("Initializing HR RAG System components...")

        # --- Initialize HuggingFace Embeddings ---
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_HF)
            print(f"Embedding model loaded: {EMBEDDING_MODEL_HF}")
        except Exception as e:
            print(f"Error loading HuggingFace Embedding model: {e}")
            print("Please ensure the model name is correct and you have internet access for the first download.")
            self.embedding_model = None # Set to None if initialization fails

        # --- Initialize ChromaDB Client and Collection ---
        # ChromaDB is used to store and retrieve vector embeddings of employee data.
        try:
            # Connect to or create the ChromaDB collection.
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_model, # Link the embedding model to the vector store
                persist_directory=CHROMA_PATH # Specify the directory for persistent storage
            )
            # Check if the vector store is empty, indicating data ingestion might be needed.
            if self.vectorstore._collection.count() == 0:
                print("Warning: ChromaDB is empty. Please run the data ingestion script first to populate it.")
            else:
                print(f"ChromaDB initialized with {self.vectorstore._collection.count()} documents.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            print("Ensure 'chroma_db_langchain' directory exists and contains valid data, or create it if new.")
            self.vectorstore = None # Set to None if initialization fails

        # --- Initialize LLM with Ollama ---
        # Ollama allows running large language models locally.
        try:
            self.llm = Ollama(model=OLLAMA_MODEL, temperature=0.1) # temperature controls creativity (lower = more focused)
            # Perform a quick test to ensure the Ollama LLM is reachable and functional.
            test_response = self.llm.invoke("Hello")
            print(f"Ollama LLM ({OLLAMA_MODEL}) initialized. Test response: {test_response[:30]}...")
        except Exception as e:
            print(f"Error initializing Ollama LLM: {e}")
            print(f"Please ensure Ollama is installed, running, and the model '{OLLAMA_MODEL}' is pulled (`ollama pull {OLLAMA_MODEL}`).")
            self.llm = None # Set to None if initialization fails

        # --- Define the Prompt Template for the LLM ---
        # This template structures the input provided to the LLM, including retrieved context and user query.
        prompt_template = PromptTemplate.from_template(
            (
                "You are an intelligent HR assistant. "
                "Based on the following employee information, answer the user's query comprehensively. "
                "If the information is not sufficient to answer, state that you don't have enough data. "
                "Always try to provide names of relevant employees and their key attributes.\n\n"
                "Context: {context}\n\n" # Placeholder for retrieved employee information
                "Question: {question}\n" # Placeholder for the user's query
                "Answer:"
            )
        )

        # --- Create a Retriever from the Vector Store ---
        # The retriever is responsible for fetching relevant documents (employee profiles)
        # from the vector store based on a query.
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr", # "similarity" for standard similarity search, "mmr" for Maximal Marginal Relevance (diversity)
                search_kwargs={"k": 5} # Retrieve the top 5 most relevant documents
            )
            print("Retriever initialized.")
        else:
            print("Retriever not initialized due to missing vectorstore.")

        # --- Construct the RAG Chain using LangChain Expression Language (LCEL) ---
        # The RAG chain orchestrates the flow: retrieve -> format -> prompt -> LLM -> parse.
        if self.retriever and self.llm:
            self.rag_chain = (
                # 1. Prepare context and question:
                #    'context' comes from the retriever, formatted by _format_docs.
                #    'question' comes directly from the input using RunnablePassthrough.
                {"context": self.retriever | RunnableLambda(self._format_docs), "question": RunnablePassthrough()}
                | prompt_template # 2. Apply the defined prompt template.
                | self.llm # 3. Pass the structured prompt to the LLM.
                | StrOutputParser() # 4. Parse the LLM's output as a simple string.
            )
            print("LangChain RAG chain constructed.")
        else:
            print("RAG chain could not be constructed due to missing retriever or LLM. Check previous error messages.")

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Formats a list of retrieved LangChain Document objects into a single string.
        This string serves as the 'context' for the LLM in the RAG chain.
        Each document's metadata (employee attributes) is extracted and presented clearly.
        """
        formatted_str = ""
        if not docs:
            return "No relevant employee information found."
        
        # Iterate through each retrieved document.
        for i, doc in enumerate(docs):
            formatted_str += f"--- Employee {i+1} ---\n"
            # Access metadata from the Document object's 'metadata' dictionary.
            # Use .get() with a default 'N/A' to prevent KeyError if an attribute is missing.
            formatted_str += f"Name: {doc.metadata.get('name', 'N/A')}\n"
            formatted_str += f"Skills: {doc.metadata.get('skills', 'N/A')}\n"
            formatted_str += f"Experience: {doc.metadata.get('experience_years', 'N/A')} years\n"
            formatted_str += f"Past Projects: {doc.metadata.get('past_projects', 'N/A')}\n"
            formatted_str += f"Availability: {doc.metadata.get('availability', 'N/A')}\n"
            formatted_str += "\n"
        return formatted_str.strip() # Remove any trailing whitespace.

    async def query_chatbot(self, user_query: str) -> str:
        """
        Asynchronously invokes the constructed RAG chain with a user query.
        This is the primary method for getting a natural language response from the HR Assistant.
        """
        if not self.rag_chain:
            # Raise an error if the RAG chain wasn't successfully initialized.
            raise RuntimeError("RAG chain is not initialized. Cannot process query. Check server logs for initialization issues.")
        # Use .ainvoke() for asynchronous execution of the LangChain runnable.
        return await self.rag_chain.ainvoke(user_query)

    async def search_employees_semantic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for employees directly using the vector store's
        similarity search capabilities. This method is independent of the LLM
        and is useful for structured data retrieval based on semantic relevance.
        """
        if not self.vectorstore:
            # Raise an error if the vector store wasn't successfully initialized.
            raise RuntimeError("Vector store is not initialized. Cannot search employees. Check server logs for initialization issues.")
        
        # Perform an asynchronous similarity search in the vector store.
        # It returns documents (employee profiles) semantically similar to the query.
        retrieved_docs = await self.vectorstore.asimilarity_search(query, k=top_k)

        found_employees_data = []
        # Extract the metadata from each retrieved Document and format it into a dictionary.
        for doc in retrieved_docs:
            employee_data = doc.metadata
            # Construct a dictionary matching the expected format of the Employee Pydantic model.
            found_employees_data.append({
                "name": employee_data.get('name', 'N/A'),
                "skills": employee_data.get('skills', 'N/A'),
                "experience_years": employee_data.get('experience_years', 0), # Default to 0 for int type
                "past_projects": employee_data.get('past_projects', 'N/A'),
                "availability": employee_data.get('availability', 'N/A')
            })
        return found_employees_data
