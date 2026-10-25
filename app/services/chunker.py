"""
文本切分服务：把长文本切成适合检索的小块（chunk）

为什么需要切分？（面试高频题）
1. Embedding 模型有输入长度上限，长文本无法整体向量化
2. 检索粒度：块太大 → 命中时夹带大量无关内容，稀释答案质量
                块太小 → 语义信息被切断，"半句话"检索不准
3. 重叠（overlap）：相邻块共享一小段文本，避免关键句正好被切在边界上

本模块实现"段落优先 + 长度约束"的朴素切分：
先按段落/换行拆成小片段，再贪心地拼接到接近 chunk_size；
单段超长时按句号/标点二次切分。后续可对比 LangChain 的
RecursiveCharacterTextSplitter 做效果实验（第 5 周优化点）。
"""
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class Chunk:
    """一个文本块及其来源信息（溯源用）"""
    text: str
    source: str        # 文档文件名
    chunk_index: int   # 块在文档中的序号


def split_text(text: str, source: str,
               chunk_size: int | None = None,
               chunk_overlap: int | None = None) -> list[Chunk]:
    """把整篇文档切分为 Chunk 列表"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # 1. 先按空行/换行拆成自然段落片段
    segments = [s.strip() for s in text.splitlines() if s.strip()]

    # 2. 处理超长段落：按中文句号/英文句号二次切分
    pieces: list[str] = []
    for seg in segments:
        if len(seg) <= chunk_size:
            pieces.append(seg)
        else:
            pieces.extend(_split_long_segment(seg, chunk_size))

    # 3. 贪心拼接：尽可能把短段落拼成一个块，块间保留 overlap
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) + 1 <= chunk_size:
            current = f"{current}\n{piece}" if current else piece
        else:
            if current:
                chunks.append(current)
            # overlap：从上一块末尾取一段拼到新块开头
            tail = current[-chunk_overlap:] if chunk_overlap and current else ""
            current = f"{tail}\n{piece}" if tail else piece
    if current:
        chunks.append(current)

    return [Chunk(text=c, source=source, chunk_index=i) for i, c in enumerate(chunks)]


def _split_long_segment(seg: str, chunk_size: int) -> list[str]:
    """超长段落按句号切分成句，再把句子拼回 chunk_size 以内的片段"""
    import re
    sentences = re.split(r"(?<=[。！？.!?])\s*", seg)
    result: list[str] = []
    buf = ""
    for sent in sentences:
        if not sent:
            continue
        if len(buf) + len(sent) <= chunk_size:
            buf += sent
        else:
            if buf:
                result.append(buf)
            # 单句本身超长时只能硬切（极端情况兜底）
            while len(sent) > chunk_size:
                result.append(sent[:chunk_size])
                sent = sent[chunk_size:]
            buf = sent
    if buf:
        result.append(buf)
    return result
