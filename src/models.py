from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatQuery(BaseModel):
    """
    Represents a user's query for the HR Assistant chatbot.
    """
    query: str = Field(..., min_length=3, description="The user's query for the HR assistant.")

class ChatResponse(BaseModel):
    """
    Represents the HR Assistant's response to a chat query.
    """
    response: str = Field(..., description="The HR assistant's answer.")

class Employee(BaseModel):
    """
    Represents a single employee's profile information.
    Note: Skills and Past Projects are strings here because they are joined
    into strings when stored in ChromaDB metadata.
    """
    name: str
    skills: str
    experience_years: int
    past_projects: str
    availability: str

class EmployeeSearchResponse(BaseModel):
    """
    Represents the response for an employee search query, containing a list of employees.
    """
    employees: List[Employee]
    message: str = "Search completed."
