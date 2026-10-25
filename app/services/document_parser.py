"""
文档解析服务：把 PDF / Word / Markdown / TXT 统一转成纯文本

设计要点：
- 每种格式一个私有函数，主入口按扩展名分发（策略模式的朴素实现）
- 统一返回 str，方便下游切分模块无差别处理
"""
from pathlib import Path

import pymupdf  # PyMuPDF
from docx import Document as DocxDocument


def parse_document(file_path: str | Path) -> str:
    """解析文档，返回纯文本。不支持的格式抛 ValueError。"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".md": _parse_text,
        ".txt": _parse_text,
    }
    parser = parsers.get(suffix)
    if parser is None:
        raise ValueError(f"不支持的文件格式: {suffix}（支持 PDF/DOCX/MD/TXT）")

    text = parser(path)
    if not text.strip():
        raise ValueError(f"文档内容为空或解析失败: {path.name}")
    return text


def _parse_pdf(path: Path) -> str:
    """逐页提取 PDF 文本，页与页之间用换行分隔"""
    parts: list[str] = []
    with pymupdf.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def _parse_docx(path: Path) -> str:
    """提取 Word 文档所有段落文本"""
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_text(path: Path) -> str:
    """直接读取文本文件（MD/TXT）"""
    return path.read_text(encoding="utf-8", errors="ignore")
