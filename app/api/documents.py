"""
文档管理路由：上传、列表

上传处理流水线（这就是 RAG 的"R"的前半段）：
  接收文件 → 保存 → 解析为纯文本 → 切分成块 → 向量化 → 存入向量库
"""
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.services import vector_store
from app.services.chunker import split_text
from app.services.document_parser import parse_document
from app.services.embedding import embed_documents

router = APIRouter(prefix="/api/documents", tags=["文档管理"])

ALLOWED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/upload", summary="上传文档并入库")
async def upload_document(file: UploadFile = File(...)):
    """上传一个文档，解析切分后向量化入库"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"不支持的格式 {suffix}，仅支持: {', '.join(sorted(ALLOWED_SUFFIXES))}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "文件超过 20MB 限制")

    started = time.time()

    # 1. 保存原始文件（保留上传记录，也方便调试）
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    saved_path = upload_dir / saved_name
    saved_path.write_bytes(content)

    # 2. 解析 → 3. 切分 → 4. 向量化 → 5. 入库
    try:
        text = parse_document(saved_path)
        chunks = split_text(text, source=file.filename)
        vectors = embed_documents([c.text for c in chunks])
        vector_store.add_chunks(chunks, vectors)
    except ValueError as e:
        saved_path.unlink(missing_ok=True)  # 解析失败则清理坏文件
        raise HTTPException(400, str(e)) from e

    return {
        "message": "文档入库成功",
        "filename": file.filename,
        "chunks": len(chunks),
        "total_chunks_in_db": vector_store.count(),
        "elapsed_seconds": round(time.time() - started, 2),
    }


@router.get("", summary="列出已入库文档")
async def list_documents():
    return {"documents": vector_store.list_sources(), "total_chunks": vector_store.count()}
