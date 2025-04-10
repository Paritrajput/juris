from fastapi import FastAPI, Request
from pydantic import BaseModel
from weather import search_faiss, build_prompt, run_reflexion  # Use your existing functions

app = FastAPI()

class UserQuery(BaseModel):
    case_description: str

@app.post("/api/analyze")
def analyze_case(query: UserQuery):
    similar = search_faiss(query.case_description, top_k=5)
    prompt = build_prompt(query.case_description, similar)
    _, improved = run_reflexion(prompt)
    return {"advice": improved, "similar_cases": similar}
