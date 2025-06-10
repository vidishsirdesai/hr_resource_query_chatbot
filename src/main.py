import uvicorn
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # Import HTMLResponse to serve HTML content
from fastapi.staticfiles import StaticFiles # Import StaticFiles to serve static assets like CSS/JS
from typing import List

# Import custom data models and the RAG (Retrieval Augmented Generation) system
from models import ChatQuery, ChatResponse, Employee, EmployeeSearchResponse
from rag_system import HRRAGSystem

# --- FastAPI Application Initialization ---
# Create a FastAPI application instance.
# Add metadata like title, description, and version for API documentation (Swagger UI/ReDoc).
app = FastAPI(
    title="HR Assistant Chatbot API",
    description="API for an intelligent HR assistant chatbot that helps find employees.",
    version="1.0.0",
)

# --- CORS (Cross-Origin Resource Sharing) Configuration ---
# Add CORS middleware to allow requests from any origin.
# This is crucial for frontend applications running on a different domain/port
# to interact with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True, # Allows credentials (cookies, HTTP authentication)
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- End CORS Configuration ---

# Initialize the HR RAG System globally.
# This ensures that the RAG system, which likely loads models or data,
# is set up once when the application starts.
hr_rag_system = HRRAGSystem()

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse, summary="Chat with the HR Assistant")
async def chat_with_hr_assistant(chat_query: ChatQuery):
    """
    **Endpoint to chat with the HR Assistant chatbot.**

    Sends a natural language query to the HR Assistant chatbot and receives a detailed response
    based on the underlying employee data and RAG capabilities.

    Args:
        chat_query (ChatQuery): A Pydantic model containing the user's query string.

    Returns:
        ChatResponse: A Pydantic model containing the chatbot's generated response.

    Raises:
        HTTPException:
            - 500 Internal Server Error if there's an issue with the chatbot system
              (e.g., initialization) or an unexpected error.
    """
    try:
        # Call the RAG system's query_chatbot method with the user's query.
        response = await hr_rag_system.query_chatbot(chat_query.query)
        # Return the response wrapped in the ChatResponse Pydantic model.
        return ChatResponse(response=response)
    except RuntimeError as re:
        # Catch specific RuntimeError for system-level issues (e.g., model loading failures).
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot system error: {re}. Please check server logs for initialization issues."
        )
    except Exception as e:
        # Catch any other unexpected exceptions during chat processing.
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing your chat query: {e}"
        )

@app.get("/employees/search", response_model=EmployeeSearchResponse, summary="Search for Employees")
async def search_employees(
    # Define query parameters for employee search.
    # 'query' is mandatory, must have a minimum length of 3, and includes a description for docs.
    query: str = Query(..., min_length=3, description="Keywords for employee search (e.g., 'Python developer', 'available for new projects', 'experience in AWS')."),
    # 'top_k' is optional with a default of 5, must be between 1 and 20, and includes a description.
    top_k: int = Query(5, ge=1, le=20, description="Number of top relevant employees to retrieve.")
):
    """
    **Endpoint to search for employees using semantic search.**

    Searches for employees based on skills, experience, past projects, or availability
    using semantic similarity, rather than exact keyword matching. Returns a list of
    the most relevant employee profiles.

    Args:
        query (str): The search query string.
        top_k (int): The maximum number of relevant employees to return.

    Returns:
        EmployeeSearchResponse: A Pydantic model containing a list of found Employee profiles.

    Raises:
        HTTPException:
            - 500 Internal Server Error if there's an issue with the employee search system
              or an unexpected error.
    """
    try:
        # Call the RAG system's semantic search method.
        found_employees_data = await hr_rag_system.search_employees_semantic(query, top_k)
        # Convert the raw employee data (e.g., dictionaries) into Pydantic Employee models.
        employees_pydantic = [Employee(**emp_data) for emp_data in found_employees_data]
        # Return the list of Pydantic employee models wrapped in the EmployeeSearchResponse.
        return EmployeeSearchResponse(employees=employees_pydantic)
    except RuntimeError as re:
        # Catch specific RuntimeError for system-level issues (e.g., data loading failures).
        raise HTTPException(
            status_code=500,
            detail=f"Employee search system error: {re}. Please check server logs for initialization issues."
        )
    except Exception as e:
        # Catch any other unexpected exceptions during employee search.
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during employee search: {e}"
        )

# --- Serve Static Files (Frontend Assets) ---
# Mount the 'static' directory to serve static files.
# Files in 'static' will be accessible via paths starting with '/static/'.
# For example, if you have 'static/css/style.css', it will be served at '/static/css/style.css'.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the 'index.html' file at the root URL ("/").
# This route uses HTMLResponse to directly return the content of an HTML file.
# `include_in_schema=False` hides this endpoint from the OpenAPI (Swagger/ReDoc) documentation,
# as it's typically for serving the frontend, not a REST API endpoint.
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """
    **Serves the main frontend application HTML file at the root URL.**

    This allows users to access the web-based UI by navigating to the base URL of the API.
    """
    # Open and read the content of the 'index.html' file located in the 'static' directory.
    # It's important that 'index.html' exists inside the 'static' folder.
    with open("static/index.html", "r") as f:
        html_content = f.read()
    # Return the HTML content as an HTMLResponse.
    return HTMLResponse(content=html_content)

# --- Main execution block for Uvicorn ---
# This block runs the FastAPI application using Uvicorn when the script is executed directly.
if __name__ == "__main__":
    print("Starting FastAPI HR Assistant API...")
    # Run the FastAPI app.
    # `host="0.0.0.0"` makes the server accessible from any IP address (useful in Docker/cloud).
    # `port=8000` sets the listening port to 8000.
    uvicorn.run(app, host="0.0.0.0", port=8000)
