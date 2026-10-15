from typing import List, Optional

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: str
    content: str


class RagRequest(BaseModel):
    question: str = ""
    history: List[Turn] = Field(default_factory=list)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    source: Optional[str] = None
    model: Optional[str] = None
    temperature: float = Field(0.1, ge=0, le=1.5)


class QueryRequest(BaseModel):  # the original class API: POST /rag/query
    question: str = Field(default="")
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    source: Optional[str] = None


class IngestFolderRequest(BaseModel):
    reset: bool = False
