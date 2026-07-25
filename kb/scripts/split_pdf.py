"""
PDF 拆分工具 —— 从源 PDF 中提取指定页码范围，生成新的 PDF 文件。
"""

import fitz
from pathlib import Path


def extract_pages(
    input_path: str,
    start_page: int,
    end_page: int,
    output_path: str,
):
    """
    从 PDF 中提取指定页码范围，保存为新文件。

    Args:
        input_path:  源 PDF 文件路径
        start_page:  起始页码（1-based，包含）
        end_page:    结束页码（1-based，包含）
        output_path: 输出 PDF 文件路径
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")

    doc = fitz.open(str(src))
    total = len(doc)

    # 转为 0-based 索引
    start_idx = start_page - 1
    end_idx = end_page - 1

    if start_idx < 0 or end_idx >= total:
        doc.close()
        raise ValueError(
            f"页码范围 {start_page}-{end_page} 超出文档范围 (共 {total} 页)"
        )

    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx)

    out = Path(output_path)
    # 确保输出目录存在
    out.parent.mkdir(parents=True, exist_ok=True)
    new_doc.save(str(out))

    new_doc.close()
    doc.close()

    print(f"✅ 已生成: {out}  (第 {start_page}-{end_page} 页, 共 {end_idx - start_idx + 1} 页)")


if __name__ == "__main__":
    # 从 raw/ 读取原始 PDF，提取指定页码范围 → processed/
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_FILE = BASE_DIR / "raw" / "059492-01.pdf"
    OUTPUT_FILE = BASE_DIR / "processed" / "栈和队列.pdf"

    extract_pages(
        input_path=str(INPUT_FILE),
        start_page=1,
        end_page=20,
        output_path=str(OUTPUT_FILE),
    )
