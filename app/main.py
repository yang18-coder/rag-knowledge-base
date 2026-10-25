"""
RAG 知识库问答系统 - 应用入口
阶段：第 1 周（API 骨架）
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="智能知识库问答系统 API",
    description="基于 RAG（检索增强生成）的文档问答服务",
    version="0.1.0",
)

# 允许前端跨域访问（第 4 周做界面时会用到）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段先放开，上线前收紧
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """健康检查"""
    return {"status": "ok", "service": "rag-knowledge-base", "version": "0.1.0"}


@app.get("/api/health")
def health():
    """供部署后的探活检查使用"""
    return {"healthy": True}
