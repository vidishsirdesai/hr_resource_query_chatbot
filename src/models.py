from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatQuery(BaseModel):
    """
    Represents a user's query sent to the HR Assistant chatbot.
    This Pydantic model defines the expected structure for incoming chat queries.
    """
    # The 'query' field is a string, it's required (...), must have a minimum length of 3 characters,
    # and includes a description for API documentation purposes.
    query: str = Field(..., min_length=3, description="The user's query for the HR assistant.")

class ChatResponse(BaseModel):
    """
    Represents the HR Assistant's response generated for a chat query.
    This Pydantic model defines the structure of the response sent back to the user.
    """
    # The 'response' field is a string, it's required (...), and provides a description.
    response: str = Field(..., description="The HR assistant's answer.")

class Employee(BaseModel):
    """
    Represents a single employee's profile information.
    This model is used to define the structure of individual employee records,
    especially when retrieved from a database like ChromaDB.
    """
    # Each field corresponds to an attribute of an employee.
    # Note: 'skills' and 'past_projects' are defined as 'str' because
    # they are typically stored as a single, concatenated string in vector databases
    # (like ChromaDB metadata) for easier embedding and retrieval.
    name: str
    skills: str
    experience_years: int
    past_projects: str
    availability: str

class EmployeeSearchResponse(BaseModel):
    """
    Represents the complete response structure for an employee search query.
    It contains a list of Employee profiles found and an optional message.
    """
    # 'employees' is a list where each item must conform to the 'Employee' Pydantic model.
    employees: List[Employee]
    # 'message' is an optional string field with a default value, providing additional context for the search.
    message: str = "Search completed."
