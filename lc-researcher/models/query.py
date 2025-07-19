from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Pydantic model for query request."""
    session_id: str
    question: str