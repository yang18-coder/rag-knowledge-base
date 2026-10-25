"""全局配置：从 .env 文件读取环境变量，避免密钥写死在代码里"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 应用信息
    app_name: str = "rag-knowledge-base"

    # 大模型（第 3 周使用）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # Embedding 模型（第 2 周使用，本地运行）
    # 默认指向项目内 models/ 目录的本地模型（离线可用）
    embedding_model: str = "models/bge-small-zh-v1.5"

    # 文本切分参数
    chunk_size: int = 500      # 每个文本块的目标长度（字符）
    chunk_overlap: int = 50    # 相邻块之间的重叠长度，防止语义被切断

    # 存储路径
    storage_dir: str = "storage"        # 向量库持久化目录
    upload_dir: str = "data/uploads"    # 上传文档保存目录

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
