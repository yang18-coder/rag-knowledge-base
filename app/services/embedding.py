"""
Embedding 服务：把文本转成向量（语义检索的基础）

原理（面试必讲）：
- Embedding 模型把一段文字映射成一个高维向量（本项目的 bge-small-zh 是 512 维）
- 语义相近的文本，向量在空间中的距离也相近
- 检索时：把用户问题转成向量，在向量库里找距离最近的文档块
- bge-small-zh-v1.5 是北京智源开源的中文模型，本地运行、免费、效果好

工程细节：
- 模型文件随项目存放（models/ 目录），离线可用，部署无需联网下载
- 模型在首次调用时才加载（懒加载），避免服务启动变慢
- encode 时对查询加 bge 要求的指令前缀，能提升检索效果
"""
from pathlib import Path

from app.core.config import settings

# 项目根目录（app/services/embedding.py 往上两级）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_model = None  # 模块级缓存：整个进程只加载一次模型

_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _resolve_model_path() -> str:
    """优先使用项目内本地模型；不存在时回退到 HuggingFace 在线加载"""
    local = _PROJECT_ROOT / settings.embedding_model
    if local.exists():
        return str(local)
    return settings.embedding_model


def _get_model():
    """懒加载 Embedding 模型"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_path = _resolve_model_path()
        print(f"[embedding] 正在加载模型 {model_path} ...")
        _model = SentenceTransformer(model_path)
        print("[embedding] 模型加载完成")
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """批量把文档块转向量"""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    """把用户的问题转向量（bge 系列对查询要求加指令前缀）"""
    model = _get_model()
    vector = model.encode([_QUERY_INSTRUCTION + query],
                          normalize_embeddings=True, show_progress_bar=False)
    return vector[0].tolist()
