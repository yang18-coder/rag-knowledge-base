"""问答路由：RAG 问答接口"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import rag

router = APIRouter(prefix="/api/chat", tags=["问答"])


class ChatRequest(BaseModel):
    """问答请求体"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户的问题")
    top_k: int = Field(default=4, ge=1, le=10, description="检索文档块数量")


class SourceInfo(BaseModel):
    source: str
    chunk_index: int
    distance: float


class ChatResponse(BaseModel):
    """问答响应体"""
    question: str
    answer: str
    sources: list[SourceInfo]


@router.post("", summary="知识库问答", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """基于已入库文档回答问题，返回答案与引用来源"""
    try:
        result = rag.ask(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"问答服务异常: {e}") from e
    return ChatResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
    )
