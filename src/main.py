import uvicorn
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # <--- Import HTMLResponse
from fastapi.staticfiles import StaticFiles # <--- Import StaticFiles
from typing import List

# Import models and the RAG system
from models import ChatQuery, ChatResponse, Employee, EmployeeSearchResponse
from rag_system import HRRAGSystem

# --- FastAPI Application ---
app = FastAPI(
    title="HR Assistant Chatbot API",
    description="API for an intelligent HR assistant chatbot that helps find employees.",
    version="1.0.0",
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Configuration ---

# Initialize the HR RAG System globally
hr_rag_system = HRRAGSystem()

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse, summary="Chat with the HR Assistant")
async def chat_with_hr_assistant(chat_query: ChatQuery):
    """
    Sends a natural language query to the HR Assistant chatbot and receives a detailed response
    based on employee data.
    """
    try:
        response = await hr_rag_system.query_chatbot(chat_query.query)
        return ChatResponse(response=response)
    except RuntimeError as re:
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot system error: {re}. Please check server logs for initialization issues."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing your chat query: {e}"
        )

@app.get("/employees/search", response_model=EmployeeSearchResponse, summary="Search for Employees")
async def search_employees(
    query: str = Query(..., min_length=3, description="Keywords for employee search (e.g., 'Python developer', 'available for new projects', 'experience in AWS')."),
    top_k: int = Query(5, ge=1, le=20, description="Number of top relevant employees to retrieve.")
):
    """
    Searches for employees based on skills, experience, projects, or availability
    using semantic search. Returns a list of matching employee profiles.
    """
    try:
        found_employees_data = await hr_rag_system.search_employees_semantic(query, top_k)
        employees_pydantic = [Employee(**emp_data) for emp_data in found_employees_data]
        return EmployeeSearchResponse(employees=employees_pydantic)
    except RuntimeError as re:
        raise HTTPException(
            status_code=500,
            detail=f"Employee search system error: {re}. Please check server logs for initialization issues."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during employee search: {e}"
        )

# --- Serve Static Files (Your Frontend) ---
# Mount the 'static' directory to serve static files.
# All files inside 'static' will be accessible under '/static/'.
# For example, 'static/style.css' would be at 'http://localhost:8000/static/style.css'
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the 'index.html' file at the root URL ("/")
# This route should come AFTER your API routes if there's any potential path conflict,
# but before the final StaticFiles catch-all if you were to have one.
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """
    Serves the main frontend application HTML file at the root URL.
    """
    # Make sure 'index.html' is inside the 'static' directory
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --- Main execution block for Uvicorn ---
if __name__ == "__main__":
    print("Starting FastAPI HR Assistant API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
