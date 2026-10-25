"""
向量数据库服务：基于 ChromaDB 的持久化存储与相似度检索

为什么选 ChromaDB？（面试可能问选型）
- 轻量：纯 Python，本地文件持久化，零部署成本，适合个人项目
- 专业：底层支持 HNSW 近似最近邻索引，检索是毫秒级的
- 可扩展：接口与 Milvus/Qdrant 等生产级向量库类似，后期可平滑迁移

核心概念：
- Collection 相当于"表"，存向量 + 原文 + 元数据
- query 时用余弦相似度（向量夹角）找最相近的 top-k 个块
"""
from pathlib import Path

import chromadb

from app.core.config import settings

_client = None
_collection = None

COLLECTION_NAME = "documents"


def _get_collection():
    """懒加载 ChromaDB 客户端与集合（单例）"""
    global _client, _collection
    if _collection is None:
        persist_dir = Path(settings.storage_dir) / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(persist_dir))
        # cosine：余弦相似度，文本检索的标准选择
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(chunks, vectors: list[list[float]]) -> int:
    """把文本块和对应向量写入向量库，返回写入数量"""
    if not chunks:
        return 0
    collection = _get_collection()
    collection.add(
        ids=[f"{c.source}#{c.chunk_index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    return len(chunks)


def search(query_vector: list[float], top_k: int = 4) -> list[dict]:
    """
    语义检索：返回与问题向量最相近的 top_k 个块

    返回格式: [{"text": ..., "source": ..., "chunk_index": ..., "distance": ...}]
    distance 是余弦距离（越小越相似，0 表示完全一致）
    """
    collection = _get_collection()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),
        })
    return hits


def count() -> int:
    """当前向量库中的块总数"""
    return _get_collection().count()


def list_sources() -> list[dict]:
    """列出已入库的所有文档及各自的块数量"""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    stats: dict[str, int] = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]:
        stats[meta["source"]] = stats.get(meta["source"], 0) + 1
    return [{"source": name, "chunks": num} for name, num in sorted(stats.items())]
