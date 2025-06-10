import pandas as pd
from faker import Faker
import random

# Initialize the Faker library to generate realistic-looking fake data.
fake = Faker()

def create_employee_data(num_employees):
    """
    Generates a Pandas DataFrame containing fake employee data.

    Args:
        num_employees (int): The number of fake employee records to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns for employee name, skills,
                      experience years, past projects, and availability.
    """

    employee_data = []

    # Define a comprehensive list of potential technical and soft skills.
    all_skills = [
        "Python", "Java", "JavaScript", "React", "Angular", "Vue.js", "Node.js",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "SQL", "NoSQL",
        "Machine Learning", "Deep Learning", "NLP", "Data Science", "Tableau",
        "Power BI", "DevOps", "CI/CD", "Terraform", "Ansible", "C#", ".NET",
        "Unity", "iOS Development", "Android Development", "React Native",
        "Flutter", "PHP", "Ruby on Rails", "Go", "TypeScript", "Kafka",
        "Spark", "Hadoop", "Scrum", "Agile", "Microservices", "REST APIs",
        "GraphQL", "Cybersecurity", "Blockchain", "AR/VR"
    ]

    # Define a list of common past project types for employees.
    all_past_projects = [
        "E-commerce Platform", "Healthcare Management System", "Fintech Application",
        "Social Media Analytics", "AI-powered Chatbot", "Supply Chain Optimization",
        "Cloud Migration Project", "Mobile Game Development", "CRM System",
        "Data Warehousing Solution", "IoT Dashboard", "Educational Platform",
        "Cybersecurity Audit", "Blockchain Voting System", "Augmented Reality App",
        "Natural Language Processing Engine", "Predictive Maintenance System",
        "Real-time Data Processing", "Customer Loyalty Program", "Fraud Detection System"
    ]

    # Define the possible availability statuses for an employee.
    availability_options = ["Available", "Partially Available", "Fully Booked"]

    # Loop to generate data for the specified number of employees.
    for _ in range(num_employees):

        # Generate a fake name for the employee.
        name = fake.name()

        # Randomly select between 1 and 5 skills for the employee.
        # random.sample ensures unique skills are chosen.
        num_skills = random.randint(1, 5)
        skills = random.sample(all_skills, num_skills)

        # Generate a random number of years of experience between 1 and 15.
        experience_years = random.randint(1, 15)

        # Randomly select between 1 and 3 past projects for the employee.
        # The range is (1, 3) to ensure at least one project is assigned.
        num_projects = random.randint(1, 3)
        past_projects = random.sample(all_past_projects, num_projects)

        # Randomly choose an availability status from the defined options.
        availability = random.choice(availability_options)

        # Append the generated employee data as a dictionary to the list.
        employee_data.append({
            "name": name,
            "skills": skills,
            "experience_years": experience_years,
            "past_projects": past_projects,
            "availability": availability
        })

    # Convert the list of employee dictionaries into a Pandas DataFrame.
    df = pd.DataFrame(employee_data)
    return df

# This block ensures the code runs only when the script is executed directly (not imported as a module).
if __name__ == "__main__":
    # Generate a DataFrame with 100 fake employee records.
    df_employee = create_employee_data(num_employees=100)
    
    # Save the DataFrame to a CSV file named 'employee_dataset.csv' in the 'data' directory.
    # index=False prevents Pandas from writing the DataFrame index as a column in the CSV.
    df_employee.to_csv("../data/employee_dataset.csv", index=False)
    print("Employee dataset generated and saved to ../data/employee_dataset.csv")
