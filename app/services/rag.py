"""
RAG 问答服务：检索增强生成的完整链路

流程（面试必背）：
  1. 用户问题 → Embedding 向量化
  2. 向量数据库检索 top-k 最相关文档块
  3. 把「问题 + 检索到的文档块」填入提示词模板
  4. 大模型基于给定资料生成回答（不许编造）
  5. 返回答案 + 引用来源（可溯源，缓解幻觉）

为什么 RAG 能缓解大模型幻觉？
  模型被明确约束"仅根据提供的资料回答"，且资料是真实检索结果，
  回答不再依赖模型参数里可能过时/错误的记忆。
"""
from dataclasses import dataclass, field

from app.services import llm, vector_store
from app.services.embedding import embed_query

# 检索返回的文档块数量：太少可能漏掉关键信息，太多会稀释上下文、增加成本
TOP_K = 4

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请严格遵守以下规则：

1. 仅根据下方提供的【参考资料】回答用户问题，不要使用资料之外的任何知识。
2. 回答末尾用【来源】标注引用了哪些文档。
3. 如果参考资料不足以回答问题，直接说"根据现有资料无法回答这个问题"，不要猜测和编造。
4. 回答简洁准确，用中文。"""


@dataclass
class RagAnswer:
    """一次问答的完整结果"""
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)  # 引用的文档块


def _build_context(hits: list[dict]) -> str:
    """把检索结果拼成提示词里的参考资料段落"""
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[资料{i}] 来源: {h['source']}（第{h['chunk_index']}块）\n{h['text']}")
    return "\n\n".join(parts)


def ask(question: str, top_k: int = TOP_K) -> RagAnswer:
    """执行一次完整的 RAG 问答"""
    # 1. 问题向量化 + 2. 语义检索
    query_vector = embed_query(question)
    hits = vector_store.search(query_vector, top_k=top_k)

    if not hits:
        return RagAnswer(
            question=question,
            answer="知识库还是空的，请先上传文档再提问。",
            sources=[],
        )

    # 3. 组装提示词（把资料放进 system，约束模型只依据资料回答）
    user_prompt = f"""【参考资料】
{_build_context(hits)}

【用户问题】
{question}"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 4. 生成回答
    answer = llm.chat(messages)

    # 5. 附上来源（溯源信息）
    return RagAnswer(
        question=question,
        answer=answer,
        sources=[
            {"source": h["source"], "chunk_index": h["chunk_index"], "distance": h["distance"]}
            for h in hits
        ],
    )
