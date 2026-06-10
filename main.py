# uvicorn main:app --reload

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("ENV KEY:", os.getenv("GOOGLE_API_KEY"))

app = FastAPI(
    title="Job Description Generator API (LangChain)",
    description="Generates SMART goals using Gemini via LangChain"
)

# ----------------------------
# Request Model
# ----------------------------
class JobDescriptionRequest(BaseModel):
    designation: str = Field(..., example="Senior Software Engineer")
    job_description: str = Field(..., example="FinTech blockchain payments platform")


# ----------------------------
# LangChain Model (Gemini 2.5 Flash)
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7
)

print("llm created successfully")

# ----------------------------
# Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an HR Performance Management AI Assistant.

Generate exactly 3 SMART goals for the employee.

Designation:
{designation}

Job Description:
{job_description}

Return ONLY valid JSON in this format:

{{
  "goals": [
    {{
      "title": "Goal 1",
      "description": "SMART goal description",
      "measurement": "KPI"
    }},
    {{
      "title": "Goal 2",
      "description": "SMART goal description",
      "measurement": "KPI"
    }},
    {{
      "title": "Goal 3",
      "description": "SMART goal description",
      "measurement": "KPI"
    }}
  ]
}}
""")


# ----------------------------
# Chain
# ----------------------------
chain = prompt | llm | StrOutputParser()


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "SMART Goal Generator API running with LangChain + Gemini 2.5 Flash"
    }


@app.post("/generate-goals")
async def generate_goals(request: JobDescriptionRequest):
    try:
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY not configured"
            )

        response_text = chain.invoke({
            "designation": request.designation,
            "job_description": request.job_description
        })

        return extract_json(response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ----------------------------
# JSON Extractor (same logic as yours)
# ----------------------------
def extract_json(text: str):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON found in model response")

    return json.loads(match.group())