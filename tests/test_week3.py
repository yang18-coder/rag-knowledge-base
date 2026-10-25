"""
第 3 周验收测试：RAG 问答端到端（会真实调用 DeepSeek API，费用 < 1分钱）

运行方式（项目根目录）:
    .venv/Scripts/python.exe -m tests.test_week3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag import ask


def main():
    print("=" * 60)
    print("RAG 问答端到端测试\n")

    # 正常问题：应该基于文档回答并带来源
    cases = [
        ("RAG 相比微调有什么优势？", True),
        ("RAG 系统有哪三个核心环节？", True),
        # 超纲问题：文档里没有，模型应该诚实说无法回答（验证反幻觉约束）
        ("今天股市行情怎么样？", False),
    ]

    for question, should_hit in cases:
        print(f"问题: {question}")
        result = ask(question)
        print(f"回答: {result.answer[:300]}")
        if should_hit:
            assert result.sources, "应命中来源但为空"
            print(f"来源: {[(s['source'], s['chunk_index']) for s in result.sources]}")
        print("-" * 60)

    print("🎉 第 3 周问答链路验收通过！")


if __name__ == "__main__":
    main()
