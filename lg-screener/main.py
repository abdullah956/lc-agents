import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict, TypedDict
from PyPDF2 import PdfReader

from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables (OpenAI key)
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# FastAPI app initialization
# ----------------------------
app = FastAPI()
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.3, model="gpt-4")

# ----------------------------
# ResumeScreenerState Type
# ----------------------------
class ResumeScreenerState(TypedDict):
    resumes: List[str]
    ranked_resumes: List[str]

# ----------------------------
# Resume Parsing Helper
# ----------------------------
def parse_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# ----------------------------
# LangGraph Nodes
# ----------------------------

# Step 1: Extract key information
def extract_info(state: ResumeScreenerState) -> ResumeScreenerState:
    extracted = []
    for resume_text in state["resumes"]:
        response = llm.predict(
            f"Extract the key information from this resume (skills, experience, education):\n{resume_text}"
        )
        extracted.append(response)
    return {"resumes": state["resumes"], "ranked_resumes": extracted}

# Step 2: Rank candidates
def rank_candidates(state: ResumeScreenerState) -> ResumeScreenerState:
    prompt = """
You are an AI recruiter. You have been given extracted resume summaries of candidates who applied for a Backend Engineer role.

Your task is to:
1. Rank all candidates from strongest to weakest (based on backend engineering experience, technical skills, and education).
2. Use this format:

1. Full Name – Summary of strengths and match.
2. ...
3. ...

3. If a candidate is not relevant, include them at the bottom with a short, respectful note.

Candidates:
"""
    joined = "\n\n".join(state["ranked_resumes"])
    final_prompt = prompt + joined

    response = llm.predict(final_prompt)

    return {"resumes": state["resumes"], "ranked_resumes": [response]}


# ----------------------------
# LangGraph Graph Definition
# ----------------------------
def build_resume_graph():
    builder = StateGraph(ResumeScreenerState)
    builder.add_node("extract_info", extract_info)
    builder.add_node("rank_candidates", rank_candidates)

    builder.set_entry_point("extract_info")
    builder.add_edge("extract_info", "rank_candidates")
    builder.add_edge("rank_candidates", END)

    return builder.compile()

# ----------------------------
# FastAPI Route: Upload & Rank
# ----------------------------
@app.post("/screen-resumes/")
async def screen_resumes(files: List[UploadFile] = File(...)):
    resumes_text = [parse_pdf(file) for file in files]

    graph = build_resume_graph()
    initial_state: ResumeScreenerState = {"resumes": resumes_text, "ranked_resumes": []}
    result = graph.invoke(initial_state)

    return JSONResponse(content={"ranking_result": result["ranked_resumes"][0]})