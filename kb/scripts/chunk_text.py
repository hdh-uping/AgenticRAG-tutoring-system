"""
结构感知 + 递归回退 + 小块合并 文本分块。

策略:
    1. 按 ## 标题切出逻辑小节
    2. 小节 < 100 字 → 向前合并
    3. 小节 > 500 字 → 递归拆分（\\n\\n → \\n → 。），20% 重叠
    4. 从 MinerU content_list 提取页码，绑定到每个 chunk

输出: data/chunks_{doc}.json / .jsonl / _txt/
"""

import argparse
import json
import re
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_GLOB = "kb/processed/*/*_text.md"   # 扫描所有已去图文本
OUTPUT_DIR = BASE_DIR / "kb" / "data"          # 按文档名输出，如 chunks_线性表.json

CHUNK_SIZE = 350          # 单块最大字符数
CHUNK_OVERLAP = 35        # 递归拆分时的重叠字符数（10%）
MIN_CHUNK_SIZE = 200      # 低于此值强制合并

# 递归拆分时的分隔符优先级
SEPARATORS = ["\n\n", "\n", "。", ".", "；", "，", " "]

HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
PAGE_MARKER_PATTERN = re.compile(r"<!--\s*page:(\d+)\s*-->")


# ── 预处理 ────────────────────────────────────────────────────

# MinerU 会从页眉/页脚提取重复的标题行，需过滤
NOISE_PATTERNS = [
    r"^## 数据结构\(C语言版\)\(第三版\)\(微课版\)\s*$",
    r"^数据结构\(C语言版\)\(第三版\)\(微课版\)\s*$",
]


def clean_markdown(text: str) -> str:
    """移除页眉/页脚噪音行。"""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if any(re.match(pat, stripped) for pat in NOISE_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ── 辅助函数 ────────────────────────────────────────────────────

def parse_headers(text: str) -> dict[int, str]:
    """
    扫描全文，建立行号 → 标题层级的映射。
    用于后续为每个 chunk 绑定其所属的标题路径。
    """
    header_map = {}
    for m in HEADER_PATTERN.finditer(text):
        lineno = text[: m.start()].count("\n")
        level = len(m.group(1))
        header_map[lineno] = f"{'#' * level} {m.group(2)}"
    return header_map


def split_by_h2(text: str) -> list[dict]:
    """
    按 ## 边界切分，每个 section 记录:
      - header_path: 所属标题路径（面包屑）
      - text: 该 section 完整文本
    """
    lines = text.split("\n")
    header_map = parse_headers(text)

    sections = []
    current_lines = []
    current_header = ""
    header_stack = {}  # level → 最近的标题

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 遇到 ## 标题 → 切出新 section
        if stripped.startswith("## ") and not stripped.startswith("### "):
            # 保存前一个 section
            if current_lines:
                sections.append({
                    "header_path": current_header or "## 正文",
                    "text": "\n".join(current_lines).strip(),
                })
            current_header = stripped
            current_lines = [line]
            continue

        # 追踪标题层级（为内层 ### 更新面包屑）
        m = HEADER_PATTERN.match(stripped) if stripped else None
        if m:
            level = len(m.group(1))
            header_stack[level] = stripped
            # 清理更深层级的过期标题
            for k in list(header_stack.keys()):
                if k > level:
                    del header_stack[k]

        current_lines.append(line)

    # 最后一个 section
    if current_lines:
        sections.append({
            "header_path": current_header or "## 正文",
            "text": "\n".join(current_lines).strip(),
        })

    return sections


# ── 分块核心逻辑 ────────────────────────────────────────────────

def merge_small_sections(sections: list[dict]) -> list[dict]:
    """
    向前合并太短的小节。

    合并后若超过 CHUNK_SIZE 也没关系——后续 split_long_sections 会再拆。
    所以这里只检查下限，不检查上限。
    """
    merged = []
    for sec in sections:
        if merged and len(sec["text"]) < MIN_CHUNK_SIZE:
            merged[-1]["text"] += "\n\n" + sec["text"]
            prev_h = merged[-1]["header_path"]
            curr_h = sec["header_path"]
            if curr_h != prev_h:
                merged[-1]["header_path"] = f"{prev_h} › {curr_h}"
        else:
            merged.append(sec.copy())
    return merged


def recursive_split_text(text: str, size: int, overlap: int) -> list[str]:
    """
    递归拆分长文本: 依次尝试分隔符，直到每段 ≤ size。
    相邻段之间保留 overlap 字符重叠。
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    for sep in SEPARATORS:
        parts = text.split(sep)
        if len(parts) > 1:
            break
    else:
        # 没有任何分隔符可用，硬切
        parts = [text[i:i+size] for i in range(0, len(text), size - overlap)]

    chunks = []
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            # 重叠: 下一段从当前段末尾 overlap 字符开始
            if len(current) > overlap:
                current = current[-overlap:] + (sep if current else "") + part
            else:
                current = (sep if current else "") + part

            # 如果单段超过 size，递归拆
            if len(current) > size:
                sub_chunks = recursive_split_text(current, size, overlap)
                if sub_chunks:
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1]
    if current.strip():
        chunks.append(current)

    return chunks


def split_long_sections(sections: list[dict]) -> list[dict]:
    """将超长 section 递归拆分，保留 header_path。"""
    result = []
    for sec in sections:
        text = sec["text"]
        if len(text) <= CHUNK_SIZE:
            result.append(sec)
        else:
            sub_texts = recursive_split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, sub in enumerate(sub_texts):
                result.append({
                    "header_path": sec["header_path"],
                    "text": sub.strip(),
                    "sub_index": i,
                })
    return result


# ── 页码提取（从 MinerU content_list 映射到 chunk）───────────────

def build_page_map(content_json_path: Path) -> dict[str, int]:
    """
    从 MinerU 的 content_list 构建文本签名 → 页码的映射。

    为避免文本被拆分后后半段匹配不到前半段的签名，每个 content 元素生成
    多个锚点签名（开头 / 中间 / 结尾各取一段），任一个命中即可定页码。
    """
    if not content_json_path.exists():
        return {}

    content = json.loads(content_json_path.read_text(encoding="utf-8"))
    page_map = {}

    for item in content:
        text = (item.get("text") or item.get("code_body") or "").strip()
        if not text or len(text) < 4:
            continue

        page_idx = item.get("page_idx", 0)

        # 每 100 字设一个锚点，保证任意 100 字切片至少命中一个锚点
        # 对于 1400 字的代码块 → 14 个锚点，无论被切成几块都能匹配
        anchors = []
        step = 100
        for start in range(0, len(text) - 4, step):
            sig = text[start:start+60].replace("\n", " ").replace("  ", " ").strip()
            if len(sig) >= 4:
                anchors.append(sig)
        # 末尾锚点（确保最后一段也能匹配）
        if len(text) > 4:
            last_sig = text[-min(60, len(text)):].replace("\n", " ").replace("  ", " ").strip()
            if len(last_sig) >= 4:
                anchors.append(last_sig)

        for sig in set(anchors):
            page_map[sig] = page_idx

        # 标题额外生成 markdown 格式签名（# 前缀）
        level = item.get("text_level")
        if level and 1 <= level <= 3:
            md_prefix = "#" * level + " "
            md_text = md_prefix + text
            for start in range(0, min(len(md_text), 60), 60):
                sig = md_text[start:start+60].replace("\n", " ").replace("  ", " ").strip()
                if len(sig) >= 4:
                    page_map[sig] = page_idx

    return page_map


def extract_inline_page_map(text: str) -> tuple[str, dict[str, int]]:
    """解析 ``<!-- page:N -->`` 标记，为无 MinerU JSON 的文本建立页码锚点。"""
    matches = list(PAGE_MARKER_PATTERN.finditer(text))
    if not matches:
        return text, {}

    page_map = {}
    for index, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        normalized = segment.replace("\n", " ").replace("  ", " ").strip()
        for anchor_start in range(0, max(len(normalized) - 4, 1), 100):
            signature = normalized[anchor_start:anchor_start + 60].strip()
            if len(signature) >= 4:
                page_map[signature] = page_num - 1

    return PAGE_MARKER_PATTERN.sub("", text), page_map


def find_page_num(chunk_text: str, page_map: dict[str, int]) -> int:
    """
    在 page_map 中搜索与 chunk 文本匹配的签名，返回页码（1-based）。

    注意：page_map 中的签名已做了换行/空格归一化处理，所以搜索前
    需要把 chunk_text 也做同样的归一化，否则签名和 chunk 格式不一致。
    """
    # 归一化 chunk_text（与 build_page_map 中的签名处理方式一致）
    normalized = chunk_text.replace("\n", " ").replace("  ", " ").strip()

    best_page = -1
    best_pos = float("inf")

    for sig, page_idx in page_map.items():
        pos = normalized.find(sig)
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_page = page_idx

    return best_page + 1 if best_page >= 0 else 0


def add_chunk_metadata(chunks: list[dict], page_map: dict[str, int] = None, doc_name: str = "") -> list[dict]:
    """
    为每个 chunk 分配 ID、统计字符数、提取页码。

    Args:
        chunks:    分块列表
        page_map:  文本签名→页码映射（由 build_page_map 构建），可选
        doc_name:  文档名，用作 chunk ID 前缀以避免跨文档主键冲突
    """
    if page_map is None:
        page_map = {}

    prefix = f"{doc_name}_" if doc_name else ""
    result = []
    for i, chk in enumerate(chunks, 1):
        page_num = find_page_num(chk["text"], page_map)
        result.append({
            "id": f"{prefix}chunk_{i:03d}",
            "header_path": chk["header_path"],
            "text": chk["text"],
            "char_count": len(chk["text"]),
            "page_num": page_num,
        })
    return result


def final_merge_small(chunks: list[dict]) -> list[dict]:
    """
    最终扫描：向前合并仍不足最小值的碎片块。

    允许合并后略超 CHUNK_SIZE（10% 容差），因为碎片块对检索的伤害
    远大于一个略超标的块。
    """
    MAX_MERGED = int(CHUNK_SIZE * 1.1)  # 550，合并容差

    merged = []
    for chk in chunks:
        if (
            merged
            and len(chk["text"]) < MIN_CHUNK_SIZE
            and len(merged[-1]["text"]) + len(chk["text"]) <= MAX_MERGED
        ):
            prev = merged[-1]
            prev["text"] += "\n\n" + chk["text"]
            prev["char_count"] = len(prev["text"])
            if chk["header_path"] != prev["header_path"]:
                prev["header_path"] += f" › {chk['header_path']}"
        else:
            merged.append(chk.copy())

    # 文档开头只有一级标题时，首块无法“向前”合并，改为并入下一块。
    if (
        len(merged) > 1
        and len(merged[0]["text"]) < MIN_CHUNK_SIZE
        and len(merged[0]["text"]) + len(merged[1]["text"]) <= MAX_MERGED
    ):
        first = merged.pop(0)
        merged[0]["text"] = first["text"] + "\n\n" + merged[0]["text"]
        merged[0]["char_count"] = len(merged[0]["text"])
        if first["header_path"] != merged[0]["header_path"]:
            merged[0]["header_path"] = f"{first['header_path']} › {merged[0]['header_path']}"
    return merged


# ── 主流程 ──────────────────────────────────────────────────────

def process_one_file(text_md_path: Path) -> dict:
    """
    处理单个文档：直接从 _text.md 分块（图片已被 describe_images.py 替换为文字描述）。
    页码通过 content.json 文本锚点匹配（锚点密度足够覆盖图片描述造成的文本偏移）。
    """
    doc_name = text_md_path.stem.replace("_text", "")
    doc_dir = text_md_path.parent

    # _text.md：图片已由 VLM 替换为文字描述，直接分块即可
    text_md = text_md_path.read_text(encoding="utf-8")
    text_md, inline_page_map = extract_inline_page_map(text_md)
    clean_text = clean_markdown(text_md)
    removed = len(text_md) - len(clean_text)

    # 页码映射（content.json 锚点密度 100 字/个，足够跨越图片描述区域）
    content_json = doc_dir / f"{doc_name}_content.json"
    page_map = build_page_map(content_json) or inline_page_map

    print(f"\n📄 {text_md_path.relative_to(BASE_DIR)}")
    print(f"   文档: {doc_name}  |  {len(clean_text):,} 字符  (清洗 {removed} 字符)")
    if page_map:
        pages_found = len(set(page_map.values()))
        print(f"   页码: {len(page_map)} 条签名 → {pages_found} 页")

    sections = split_by_h2(clean_text)
    sections = merge_small_sections(sections)
    chunks = split_long_sections(sections)
    chunks = add_chunk_metadata(chunks, page_map, doc_name)
    chunks = final_merge_small(chunks)
    for i, chk in enumerate(chunks, 1):
        chk["id"] = f"{doc_name}_chunk_{i:03d}"

    sizes = [c["char_count"] for c in chunks]
    print(f"   分块: {len(chunks)} 个  |  平均 {sum(sizes)//len(sizes):.0f} 字  |  范围 {min(sizes)}-{max(sizes)} 字")

    return {
        "doc_name": doc_name,
        "chunks": chunks,
        "stats": {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "median_size": sorted(sizes)[len(sizes) // 2],
        },
    }


def save_chunks(result: dict, output_dir: Path):
    """按文档名保存分块结果，支持多种格式。"""
    doc_name = result["doc_name"]
    chunks = result["chunks"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON（完整数组，适合程序读取）
    json_path = output_dir / f"chunks_{doc_name}.json"
    json_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # JSONL（每行一个 chunk，适合流式加载、大批量）
    jsonl_path = output_dir / f"chunks_{doc_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chk in chunks:
            f.write(json.dumps(chk, ensure_ascii=False) + "\n")

    # 独立 txt 目录（每个 chunk 一个文件，方便直接查看）
    txt_dir = output_dir / f"chunks_{doc_name}_txt"
    txt_dir.mkdir(exist_ok=True)
    expected_txt_names = {f"{chk['id']}.txt" for chk in chunks}
    for stale_path in txt_dir.glob(f"{doc_name}_chunk_*.txt"):
        if stale_path.name not in expected_txt_names:
            stale_path.unlink()
    for chk in chunks:
        txt_path = txt_dir / f"{chk['id']}.txt"
        # 第一行写 header_path 作为注释，之后写正文
        txt_path.write_text(
            f"# {chk['header_path']}\n\n{chk['text']}",
            encoding="utf-8",
        )

    def display_path(path: Path) -> Path:
        try:
            return path.relative_to(BASE_DIR)
        except ValueError:
            return path

    print(f"   ✅ {display_path(json_path)}")
    print(f"   ✅ {display_path(jsonl_path)}")
    print(f"   ✅ {display_path(txt_dir)}/  ({len(chunks)} 个 txt)")


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="只处理指定文档，例如：数组")
    parser.add_argument("--text-file", type=Path, help="直接处理指定 _text.md")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="分块输出目录（默认 kb/data）",
    )
    args = parser.parse_args()

    md_files = (
        [args.text_file.resolve()]
        if args.text_file else sorted(BASE_DIR.glob(INPUT_GLOB))
    )
    if args.doc and not args.text_file:
        md_files = [
            path for path in md_files
            if path.parent.name == args.doc and path.stem == f"{args.doc}_text"
        ]
    if not md_files:
        print(f"❌ 未找到文件: {INPUT_GLOB}")
        return

    print(f"📂 扫描到 {len(md_files)} 个待分块文件")

    all_stats = []
    for md_path in md_files:
        result = process_one_file(md_path)
        save_chunks(result, args.output_dir)
        all_stats.append(result["stats"])

    print(f"\n{'='*60}")
    try:
        output_label = args.output_dir.relative_to(BASE_DIR)
    except ValueError:
        output_label = args.output_dir
    print(f"🎉 全部完成: {len(md_files)} 个文档 → {output_label}/")
    for s in all_stats:
        print(f"   chunks_{md_files[all_stats.index(s)].stem.replace('_text','')}: {s['total_chunks']} 块")


if __name__ == "__main__":
    main()
