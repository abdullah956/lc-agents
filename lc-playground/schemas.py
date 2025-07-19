from pydantic import BaseModel
from typing import List

class PromptRequest(BaseModel):
    user_input: str
    styles: List[str]

class PromptResult(BaseModel):
    style: str
    response: str
