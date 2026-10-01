import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
from google import genai
from google.genai import types

app = FastAPI(title="Synapse AI - CramIQ Triage Engine")

# 1. Enable CORS so Framer frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from your Framer prototype URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client (uses GEMINI_API_KEY environment variable)
client = genai.Client()

# 2. Define Pydantic Schema for Strict JSON Output
class ConceptItem(BaseModel):
    id: str = Field(description="Concept number, e.g. '01'")
    title: str = Field(description="Name of the high-yield concept")
    takeaway: str = Field(description="Key fact or memory hook")

class ScheduleSlot(BaseModel):
    time_window: str = Field(description="Time slot, e.g. '00:00 - 10:00'")
    action: str = Field(description="Specific recall task")
    focus: str = Field(description="Core concept focus")

class CramIQTriageResponse(BaseModel):
    topic_title: str
    time_remaining: str
    core_concepts: list[ConceptItem]
    emergency_schedule: list[ScheduleSlot]

SYSTEM_PROMPT = """
You are the CramIQ 80/20 Triage Engine for Synapse AI. 
Your job is to analyze raw study materials/notes, strip away 80% of unnecessary fluff, 
and extract ONLY the top 20% highest-yield exam concepts.
Create an actionable, high-intensity emergency study plan based on the input text.
"""

@app.get("/")
def read_root():
    return {"status": "Synapse AI Backend is Live!"}

@app.post("/api/triage", response_model=CramIQTriageResponse)
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Extract text from uploaded PDF
    try:
        reader = PdfReader(file.file)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="PDF contains no extractable text.")

    # Query Gemini model with Structured JSON Output
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\nStudy Material:\n{extracted_text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CramIQTriageResponse,
                temperature=0.2,
            )
        )
        
        # Parse and return JSON response
        result_data = json.loads(response.text)
        return result_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Triage processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)