import pandas as pd
from faker import Faker
import random

fake = Faker()

def create_employee_data(num_employees):

    employee_data = []

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

    all_past_projects = [
        "E-commerce Platform", "Healthcare Management System", "Fintech Application",
        "Social Media Analytics", "AI-powered Chatbot", "Supply Chain Optimization",
        "Cloud Migration Project", "Mobile Game Development", "CRM System",
        "Data Warehousing Solution", "IoT Dashboard", "Educational Platform",
        "Cybersecurity Audit", "Blockchain Voting System", "Augmented Reality App",
        "Natural Language Processing Engine", "Predictive Maintenance System",
        "Real-time Data Processing", "Customer Loyalty Program", "Fraud Detection System"
    ]

    availability_options = ["Available", "Partially Available", "Fully Booked"]

    for _ in range(num_employees):

        name = fake.name()

        # Ensure at least 1 skill
        num_skills = random.randint(1, 5)
        skills = random.sample(all_skills, num_skills)

        experience_years = random.randint(1, 15)

        # Ensure at least 1 past project
        # Change the range from (0, 3) to (1, 3) or (1, len(all_past_projects))
        # to guarantee at least one project.
        num_projects = random.randint(1, 3) # Changed from (0, 3)
        past_projects = random.sample(all_past_projects, num_projects)

        availability = random.choice(availability_options)

        employee_data.append({
            "name": name,
            "skills": skills,
            "experience_years": experience_years,
            "past_projects": past_projects,
            "availability": availability
        })

    df = pd.DataFrame(employee_data)
    return df
        

if __name__ == "__main__":
    df_employee = create_employee_data(num_employees = 100)
    # adding index = False to prevent "Unnamed: 0"
    df_employee.to_csv("../data/employee_dataset.csv", index=False)
