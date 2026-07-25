"""
使用 MinerU API 批量解析 PDF 文档，输出 Markdown 及关联资源。

数据流:
    raw/*.pdf  →  utils/split_pdf.py（去目录、去习题）
               →  processed/*.pdf
               →  build_kb/parse_pdf.py（MinerU 解析）
               →  processed/{文件名}/（md, html, docx, json, images/）

API 文档: https://mineru.net/doc/docs/index.html

用法:
    conda activate tec_stack
    python build_kb/parse_pdf.py
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mineru import MinerU

# ── 配置 ────────────────────────────────────────────────────────
# 项目根目录（脚本在 kb/scripts/ 下，向上两级到项目根）
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# 源文件目录（经过 split 清洗后的 PDF）
INPUT_DIR = BASE_DIR / "kb" / "processed"

# 解析输出目录（MinerU 结果，按文件名保存在 processed 子目录下）
OUTPUT_DIR = BASE_DIR / "kb" / "processed"

# MinerU API Token 只能通过环境变量提供。
TOKEN = os.getenv("MINERU_TOKEN", "")

# 解析参数
PARSE_PARAMS = {
    "model": "vlm",           # 模型: pipeline / vlm / html (vlm 精度最高)
    "ocr": True,              # 开启 OCR（扫描件识别）
    "language": "ch",         # 文档语言：中文
    "formula": True,          # 公式识别
    "table": True,            # 表格识别
    "timeout": 1800,          # 单文件超时 30 分钟
}

# ── 辅助函数 ────────────────────────────────────────────────────

def collect_pdf_files(input_dir: Path, doc_name: str = "") -> list[Path]:
    """收集待解析的 PDF 文件列表。"""
    if not input_dir.exists():
        raise FileNotFoundError(f"源文件目录不存在: {input_dir}")

    if doc_name:
        target = input_dir / f"{doc_name}.pdf"
        pdf_files = [target] if target.exists() else []
    else:
        pdf_files = sorted(input_dir.glob("*.pdf"))  # 只取根目录 PDF，不递归子目录
    if not pdf_files:
        print(f"⚠️  {input_dir} 目录下没有 PDF 文件")
        sys.exit(0)

    print(f"📂 扫描到 {len(pdf_files)} 个 PDF 文件:")
    for f in pdf_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name}  ({size_mb:.1f} MB)")
    return pdf_files


def save_result(result, output_dir: Path) -> Path | None:
    """
    保存单个文档的解析结果。每种格式独立保存，单个失败不影响其他。

    MinerU 返回的 ExtractResult 包含:
      - markdown:  Markdown 文本
      - content_list: 结构化内容列表（段落、表格、图片等）
      - images:     解析出的图片列表
      - html / docx / latex: 需请求 extra_formats 才有

    只要 Markdown 保存成功即视为有效。
    """
    stem = Path(result.filename).stem if result.filename else "unknown"
    doc_dir = output_dir / stem
    doc_dir.mkdir(parents=True, exist_ok=True)

    saved = False

    # 1. Markdown — 核心产出，必须有
    if result.markdown:
        try:
            md_path = result.save_markdown(str(doc_dir / f"{stem}.md"), with_images=True)
            print(f"   ✅ Markdown: {len(result.markdown)} 字符 → {md_path}")
            saved = True
        except Exception as e:
            print(f"   ❌ Markdown 保存失败: {e}")
    else:
        print(f"   ❌ Markdown 为空")

    # 2. HTML — 需要 extra_formats=["html"]
    if result.html:
        try:
            html_path = result.save_html(str(doc_dir / f"{stem}.html"))
            print(f"   ✅ HTML:     {html_path}")
        except Exception as e:
            print(f"   ⚠️  HTML 保存失败: {e}")

    # 3. DOCX — 需要 extra_formats=["docx"]
    if result.docx:
        try:
            docx_path = result.save_docx(str(doc_dir / f"{stem}.docx"))
            print(f"   ✅ DOCX:     {docx_path}")
        except Exception as e:
            print(f"   ⚠️  DOCX 保存失败: {e}")

    # 4. content_list JSON（结构化数据，便于后续分块等处理）
    if result.content_list:
        try:
            import json
            content_path = doc_dir / f"{stem}_content.json"
            content_path.write_text(
                json.dumps(result.content_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"   ✅ Content:  {len(result.content_list)} 个元素")
        except Exception as e:
            print(f"   ⚠️  Content 保存失败: {e}")

    return doc_dir if saved else None


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="使用 MinerU 解析 processed 目录中的 PDF")
    parser.add_argument("--doc", default="", help="只解析指定文档名，例如：数组")
    parser.add_argument("--pdf", type=Path, help="直接解析指定 PDF 文件")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="解析产物根目录"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MinerU 批量文档解析")
    print("=" * 60)

    # 1. 扫描文件（processed 目录下的 PDF，即 utils 清洗后的结果）
    if args.pdf:
        pdf_path = args.pdf.resolve()
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF 不存在或格式不正确: {pdf_path}")
        pdf_files = [pdf_path]
    else:
        pdf_files = collect_pdf_files(INPUT_DIR, args.doc)

    # 2. 初始化客户端
    if not TOKEN:
        print("❌ 未设置 MINERU_TOKEN，请先设置 API Token")
        print("   获取地址: https://mineru.net/apiManage/token")
        sys.exit(1)

    client = MinerU(TOKEN)
    print(f"\n🔗 API 端点: {client._base_url if hasattr(client, '_base_url') else 'https://mineru.net/api/v4'}")

    # 3. 批量解析
    sources = [str(f) for f in pdf_files]
    print(f"\n🚀 开始批量解析 {len(sources)} 个文件...\n")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    try:
        # extract_batch 返回迭代器，文件逐个完成时就 yield
        for result in client.extract_batch(sources, **PARSE_PARAMS):
            name = result.filename or "unknown"

            if result.state == "done":
                print(f"\n📄 [{name}] 解析完成 (task: {result.task_id})")
                doc_dir = save_result(result, output_dir)
                if doc_dir:
                    success_count += 1
                else:
                    fail_count += 1
            else:
                print(f"\n📄 [{name}] 状态异常: state={result.state}, error={result.error}")
                fail_count += 1

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 批处理异常: {e}")
        import traceback
        traceback.print_exc()
        fail_count += max(1, len(sources) - success_count)

    # 4. 汇总
    print("\n" + "=" * 60)
    print(f"📊 处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print(f"📁 结果目录: {output_dir}")
    print("=" * 60)
    return 0 if success_count == len(sources) and fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
