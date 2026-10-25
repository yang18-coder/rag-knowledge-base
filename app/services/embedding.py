"""
Embedding 服务：把文本转成向量（语义检索的基础）

原理（面试必讲）：
- Embedding 模型把一段文字映射成一个高维向量（本项目的 bge-small-zh 是 512 维）
- 语义相近的文本，向量在空间中的距离也相近
- 检索时：把用户问题转成向量，在向量库里找距离最近的文档块
- bge-small-zh-v1.5 是北京智源开源的中文模型，本地运行、免费、效果好

工程细节：
- 模型在首次调用时才加载（懒加载），避免服务启动变慢
- 支持 HF_ENDPOINT 环境变量走国内镜像下载模型
- encode 时对查询加 bge 要求的指令前缀，能提升检索效果
"""
import os

from app.core.config import settings

# 国内网络环境优先走 HuggingFace 镜像（必须在 import 前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_model = None  # 模块级缓存：整个进程只加载一次模型

_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _get_model():
    """懒加载 Embedding 模型"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embedding] 正在加载模型 {settings.embedding_model}（首次加载需下载，请耐心等待）...")
        _model = SentenceTransformer(settings.embedding_model)
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
