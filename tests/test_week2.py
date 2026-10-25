"""
第 2 周验收测试：跑通「解析 → 切分 → 向量化 → 入库 → 检索」全链路

运行方式（项目根目录）:
    .venv/Scripts/python.exe -m tests.test_week2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import vector_store
from app.services.chunker import split_text
from app.services.document_parser import parse_document
from app.services.embedding import embed_documents, embed_query


def main():
    sample = Path("data/samples/RAG技术入门.md")

    print("=" * 60)
    print("步骤 1/5：解析文档")
    text = parse_document(sample)
    print(f"  ✓ 解析出 {len(text)} 个字符")

    print("步骤 2/5：文本切分")
    chunks = split_text(text, source=sample.name)
    print(f"  ✓ 切成 {len(chunks)} 个块，平均长度 {sum(len(c.text) for c in chunks) // len(chunks)} 字符")
    assert all(len(c.text) <= 600 for c in chunks), "存在超长块，切分逻辑有 bug"

    print("步骤 3/5：向量化（首次运行会自动下载模型，约 100MB）")
    vectors = embed_documents([c.text for c in chunks])
    print(f"  ✓ 生成 {len(vectors)} 个向量，维度 {len(vectors[0])}")

    print("步骤 4/5：写入向量数据库")
    vector_store.add_chunks(chunks, vectors)
    print(f"  ✓ 当前库中共 {vector_store.count()} 个块")

    print("步骤 5/5：语义检索测试")
    questions = ["RAG 相比微调有什么好处？", "文本切多大比较合适？"]
    for q in questions:
        hits = vector_store.search(embed_query(q), top_k=2)
        print(f"\n  问题: {q}")
        for h in hits:
            preview = h["text"][:50].replace("\n", " ")
            print(f"    [距离 {h['distance']}] {preview}...")

    print("\n" + "=" * 60)
    print("🎉 第 2 周全链路验收通过！")


if __name__ == "__main__":
    main()
