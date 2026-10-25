# 智能知识库问答系统（RAG）

基于 RAG（Retrieval-Augmented Generation，检索增强生成）架构的文档问答系统：
上传 PDF / Word / Markdown 文档，AI 基于文档内容精准回答问题，并标注答案来源。

> 🚧 当前状态：第 1 周 — API 骨架阶段（开发中）

## 技术栈（规划）

- **后端**：Python 3.13 + FastAPI
- **RAG**：LangChain + ChromaDB（向量数据库）
- **大模型**：DeepSeek API
- **前端**：Gradio（原型）→ Vue3（正式版）
- **部署**：Docker

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
uvicorn app.main:app --reload

# 5. 打开交互式 API 文档
# 浏览器访问 http://127.0.0.1:8000/docs
```

## 项目结构

```
rag-knowledge-base/
├── app/
│   ├── main.py          # FastAPI 应用入口
│   ├── api/             # 路由（文档上传、问答等接口）
│   ├── core/            # 配置、常量
│   └── services/        # 核心业务：文档解析、检索、问答
├── tests/               # 单元测试
├── docs/                # 项目文档
├── requirements.txt
└── README.md
```

## 开发路线

- [x] 第 1 周：FastAPI 骨架、虚拟环境、Git 规范
- [ ] 第 2 周：文档解析 + 切分 + 向量化入库
- [ ] 第 3 周：检索问答核心链路（MVP）
- [ ] 第 4 周：Web 界面
- [ ] 第 5 周：混合检索 / Rerank / 流式输出
- [ ] 第 6 周：RAGAS 评估 + Docker 部署
