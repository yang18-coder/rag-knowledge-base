"""
大模型服务：封装 DeepSeek API 调用

为什么用 openai 官方 SDK？
- DeepSeek API 完全兼容 OpenAI 协议，换 base_url 即可
- 学一套 SDK 通用所有主流模型（OpenAI/DeepSeek/通义/智谱...），面试加分

核心概念：
- messages: 对话历史列表，role 分 system(设定)/user(提问)/assistant(回复)
- temperature: 0~2，越低越稳定忠实，知识问答用 0.1 防止模型"自由发挥"
- 流式输出: stream=True 时逐字返回，前端可做打字机效果（第 4 周用）
"""
from openai import OpenAI

from app.core.config import settings

_client = None  # 模块级缓存


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _client


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    """同步调用大模型，返回完整回答文本"""
    response = _get_client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def chat_stream(messages: list[dict], temperature: float = 0.1):
    """流式调用：逐段 yield 回答片段（第 4 周前端打字机效果用）"""
    stream = _get_client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
