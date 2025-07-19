import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from prompt_styles import apply_style
from schemas import PromptRequest, PromptResult
from typing import List

import openai
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# Create OpenAI client
client = OpenAI(api_key=api_key)

# FastAPI app
app = FastAPI(title="Prompt Engineering Playground")

@app.post("/compare", response_model=List[PromptResult])
async def compare_prompts(request: PromptRequest):
    results = []
    for style in request.styles:
        styled_prompt = apply_style(style, request.user_input)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": styled_prompt}]
            )
            answer = response.choices[0].message.content.strip()
            results.append(PromptResult(style=style, response=answer))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return results
