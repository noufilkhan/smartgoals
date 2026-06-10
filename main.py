# uvicorn main:app --reload
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
import json
import re

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI(
    title="Job Description Generator API",
    description="API to generate formatted Job Descriptions based on Designation and Company Profile."
)

# Define the Pydantic model for the incoming POST request
class JobDescriptionRequest(BaseModel):
    designation: str = Field(..., example="Senior Software Engineer")
    job_description: str = Field(..., example="A fast-growing FinTech startup focusing on blockchain-based cross-border payments.")


@app.get("/")
async def root():
    return {"message": "Welcome to the SMART Goal Generator API!"}


@app.post("/generate-goals")
async def generate_goals(request: JobDescriptionRequest):

    try:
        #print(f"Received request to generate job description for designation: '{request.designation}' with company profile: '{request.company_profile}'")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured. Please set the GOOGLE_API_KEY environment variable.")

        # Construct the prompt enforcing your exact formatting rules
        prompt = f"""
            You are an HR Performance Management AI Assistant. 
            Your task is to generate exactly three SMART (Specific, Measurable, Achievable, Relevant, and Time-bound) annual performance goals for an employee based on the provided Designation and Job Description. Carefully analyze the employee's role, responsibilities, and expected business outcomes, then create three professional goals that are directly aligned with the job requirements. 
            Each goal must be specific to the employee's position, include measurable success criteria or key performance indicators (KPIs), be realistic and achievable within the employee's scope of responsibility, contribute to organizational objectives, and include a clear completion timeframe, typically within the next 12 months. 
            Focus on areas such as business impact, operational efficiency, quality improvement, customer satisfaction, innovation, technical excellence, leadership, compliance, collaboration, or professional development as appropriate for the role. Avoid generic goals that could apply to any employee and ensure the goals are tailored to the provided designation and job description.

            Designation:
            '{request.designation}'

            Job Description:
            '{request.job_description}'

            Return exactly 3 goals in the following JSON format and do not include any additional text:

            {{
                "goals": [
                    {{
                        "title": "Goal 1 Title",
                        "description": "SMART goal description",
                        "measurement": "KPI or measurable success criteria"
                    }},
                    {{
                        "title": "Goal 2 Title",
                        "description": "SMART goal description",
                        "measurement": "KPI or measurable success criteria"
                    }},
                    {{
                        "title": "Goal 3 Title",
                        "description": "SMART goal description",
                        "measurement": "KPI or measurable success criteria"
                    }}
                ]
            }}
            """
                
        # Generate the response using the LLM
        print("Generating job description with the following prompt:")

        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        #return json.loads(response.text)
        return extract_json(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def extract_json(text: str):
    text = text.strip()

    # remove markdown fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    # extract JSON object safely
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model response")

    return json.loads(match.group())
