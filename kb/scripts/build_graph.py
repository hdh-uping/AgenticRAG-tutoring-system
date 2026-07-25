"""
Knowledge Graph 构建 —— 从 MinerU content_list 抽取 → Neo4j

实体类型 (3):  数据结构 | 操作 | 复杂度
关系类型 (3):  IS_A | HAS_OPERATION | HAS_COMPLEXITY

关键改进:
  - 操作节点带数据结构前缀 ("顺序表·插入"), 不同数据结构的同名操作不会混淆
  - 操作节点携带 description / pseudocode / code, 是完整知识单元, 不依赖回查
  - 按 ## 章节边界分批, 保证操作和代码在同一批
  - 全局补全: 缺失的 code 和 HAS_COMPLEXITY 从 content_list 自动补

用法:
  python graph/build_graph.py
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

# ── 配置 ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
PROCESSED_DIR = BASE_DIR / "kb" / "processed"
OUTPUT_DIR = BASE_DIR / "kb" / "graph"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def default_content_files() -> list[Path]:
    """自动发现所有标准 MinerU content.json，避免新增文档时修改代码。"""
    return sorted(PROCESSED_DIR.glob("*/*_content.json"))

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    if not LLM_API_KEY:
        raise RuntimeError("未配置 LLM_API_KEY，无法构建知识图谱")
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=float(os.getenv("GRAPH_LLM_TIMEOUT_SECONDS", "180")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )

# ── Prompt ──────────────────────────────────────────────────────

EXTRACT_SYSTEM = """你是知识图谱构建专家。从数据结构教材的 content_list 中提取实体和关系。

## 实体类型（只能是这3种）
- 数据结构: 教材中明确出现的数据结构，例如线性表、链表、栈、队列、数组、矩阵、稀疏矩阵
- 操作: 必须以 "{数据结构名}·{操作名}" 格式命名，如 "顺序表·插入" "单链表·删除"
- 复杂度: O(n)、O(1) 等时间复杂度

## 关系类型（只能是这3种）
- IS_A: 子结构 → 父结构（顺序表 → 线性表）
- HAS_OPERATION: 数据结构 → 操作（顺序表 → 顺序表·插入）
- HAS_COMPLEXITY: 操作 → 复杂度（顺序表·插入 → O(n)）

## 每个实体必须的字段
- 数据结构: name, description(1-2句中文定义), content_ids(数组)
- 操作: name, description(操作的核心逻辑), pseudocode(1-5步算法步骤),
         code(完整C代码, 如果content_list中有), content_ids
- 复杂度: name, content_ids

## 输出格式（严格JSON, 不要markdown标记, 不要解释）
{
  "entities": [
    {"name": "顺序表", "type": "数据结构", "description": "用一组连续存储单元依次存放线性表数据元素的存储结构",
     "content_ids": ["线性表:content:0006","线性表:content:0008"]},
    {"name": "顺序表·插入", "type": "操作",
     "description": "在顺序表第i个位置前插入新元素e，将第i到第n个元素后移一位，表长+1",
     "pseudocode": "1. 检查i合法性(0≤i≤n) 2. 元素从后向前后移 3. 新元素放入第i位 4. 表长+1",
     "code": "int ListInsert_Sq(SqList *L, int i, ElemType e) { ... }",
     "content_ids": ["线性表:content:0044","线性表:content:0045"]},
    {"name": "O(n)", "type": "复杂度", "content_ids": ["线性表:content:0051"]}
  ],
  "relations": [
    {"head": "顺序表", "relation": "IS_A", "tail": "线性表"},
    {"head": "顺序表", "relation": "HAS_OPERATION", "tail": "顺序表·插入"},
    {"head": "顺序表·插入", "relation": "HAS_COMPLEXITY", "tail": "O(n)"}
  ]
}

## 规则
- content_ids 必须是输入数据中的 idx 字段值
- 操作节点 name 必须带前缀: "{数据结构名}·{操作名}"
- code 字段从 content_list 中 type=code 的条目原文提取，保留完整C代码
- 只抽取明确出现的内容，不猜测不补充"""

EXTRACT_USER = "章节: {chapter}\n\ncontent_list（JSON数组）:\n{content}\n\n请抽取实体和关系（纯JSON）:"


MERGE_SYSTEM = """你是数据清洗专家。以下是知识图谱实体的 name 和 type 列表。
请把同义名称合并，输出规则。

规则:
- 保留最短、最通用的名称作为主名
- 每个实体可以有多个 aliases
- 只能合并 type 相同的实体
- 数据结构之间不要合并：顺序表、单链表、循环链表等是不同结构，不是别名
- 复杂度之间不要合并
- 仅对“操作”判断同义，且结构前缀必须完全相同
  （如"单链表·删除节点"="单链表·删除"；"顺序表·删除"不能与"单链表·删除"合并）
- primary 和 aliases 都必须逐字来自输入，不得创造新名称
- 不确定的不合并

输出严格JSON:
{"merges": [{"primary": "顺序表", "aliases": ["顺序存储结构"]}, ...]}"""


# ── 分批: 按 ## 章节边界 ─────────────────────────────────────

def split_by_chapter(content_list: list[dict]) -> list[tuple[str, list[dict]]]:
    """
    按 content_list 中的 ## 标题(text_level=2)边界切分批次。
    每批附带章节路径作为上下文。
    """
    chapters = []
    current_title = "正文"
    current_batch = []
    # 维护章节面包屑
    stack = {1: "", 2: "", 3: ""}

    for item in content_list:
        level = item.get("text_level")
        text = (item.get("text") or "").strip()

        if level == 2 and text:
            # 新 ## 章节 → 保存上一批
            if current_batch:
                path = " > ".join(v for v in [stack[1], stack[2]] if v) or "正文"
                chapters.append((path, current_batch))
            current_batch = []
            stack[2] = text
            stack[3] = ""

        elif level == 1 and text:
            stack[1] = text
        elif level == 3 and text:
            stack[3] = text

        current_batch.append(item)

    # 最后一批
    if current_batch:
        path = " > ".join(v for v in [stack[1], stack[2]] if v) or "正文"
        chapters.append((path, current_batch))

    # 合并太小的批次
    merged = []
    buffer_title, buffer_batch = "", []
    for title, batch in chapters:
        if len(buffer_batch) + len(batch) <= 60:
            buffer_batch.extend(batch)
            buffer_title = buffer_title or title
        else:
            if buffer_batch:
                merged.append((buffer_title, buffer_batch))
            buffer_title, buffer_batch = title, batch
    if buffer_batch:
        merged.append((buffer_title, buffer_batch))

    max_items = int(os.getenv("GRAPH_BATCH_ITEMS", "25"))
    bounded = []
    for title, batch in merged:
        for start in range(0, len(batch), max_items):
            part = batch[start:start + max_items]
            suffix = f"（{start + 1}-{start + len(part)}）" if len(batch) > max_items else ""
            bounded.append((title + suffix, part))
    return bounded


# ── 提取 ──────────────────────────────────────────────────────

def extract_batch(chapter: str, batch: list[dict]) -> dict:
    """调用 LLM 抽取一批的实体和关系"""
    slim = [{"idx": c["idx"], "type": c.get("type"), "text": c.get("text") or c.get("code_body", "")}
            for c in batch]
    content_str = json.dumps(slim, ensure_ascii=False, indent=2)

    last_error = None
    for attempt in range(1, 4):
        try:
            resp = get_llm_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": EXTRACT_USER.format(chapter=chapter, content=content_str)},
                ],
                temperature=0.1,
                max_tokens=int(os.getenv("GRAPH_EXTRACT_MAX_TOKENS", "8192")),
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = (resp.choices[0].message.content or "").strip()
            return json.loads(raw)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                print(f"调用/JSON 失败，重试 {attempt}/3: {exc}", end="; ", flush=True)
                time.sleep(attempt * 2)
    raise RuntimeError(f"LLM 连续三次抽取失败: {last_error}")


def extract_batch_cached(
    chapter: str,
    batch: list[dict],
    cache_dir: Path | None,
) -> tuple[dict, bool]:
    """按输入、模型和 Prompt 哈希缓存已成功的抽取批次。"""
    if cache_dir is None:
        return extract_batch(chapter, batch), False
    cache_payload = json.dumps(
        {
            "model": LLM_MODEL,
            "system": EXTRACT_SYSTEM,
            "chapter": chapter,
            "batch": batch,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), True
    result = extract_batch(chapter, batch)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(cache_path)
    return result, False


# ── 全局补全 ──────────────────────────────────────────────────

def complete_missing(entities: list[dict], relations: list[dict],
                     content_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    补全缺失的 code 和 HAS_COMPLEXITY:
      1. 每个操作节点, 从它的 content_ids 附近搜索 code_body
      2. 每个操作节点, 从附近搜索 "时间复杂度为 O(n)" 并连到复杂度节点
    """
    # 建立 name → entity 索引
    ent_by_name = {e["name"]: e for e in entities}

    for ent in entities:
        if ent["type"] != "操作":
            continue
        cids = ent.get("content_ids", [])
        if not cids:
            continue

        # 扩展搜索窗口: content_ids 最近的那个 ±10 范围
        position_by_id = {item["idx"]: position for position, item in enumerate(content_list)}
        anchor_id = cids[len(cids)//2]  # 取中间的 content_id
        anchor = position_by_id.get(anchor_id)
        if anchor is None:
            continue
        start = max(0, anchor - 15)
        end = min(len(content_list), anchor + 15)
        neighbors = content_list[start:end]

        # 补 code
        if not ent.get("code"):
            for item in neighbors:
                code = (item.get("code_body") or "").strip()
                # 检查 code 是否和当前操作相关（操作名中的关键词出现在代码附近）
                op_keyword = ent["name"].split("·")[-1]  # "顺序表·插入" → "插入"
                if code and len(code) > 20:
                    # 看代码前后文本是否含操作关键词
                    idx = position_by_id[item["idx"]]
                    nearby_text = " ".join(
                        (c.get("text") or c.get("code_body") or "")
                        for c in content_list[max(0,idx-3):min(len(content_list),idx+3)]
                    )
                    if op_keyword in nearby_text or ent["name"].split("·")[0] in nearby_text:
                        ent["code"] = code
                        # 扩展 content_ids
                        new_ids = set(ent.get("content_ids", []))
                        for c in content_list[max(0,idx-2):min(len(content_list),idx+3)]:
                            new_ids.add(c["idx"])
                        ent["content_ids"] = sorted(new_ids)
                        break

        # 补 HAS_COMPLEXITY
        already_has = any(
            r["head"] == ent["name"] and r["relation"] == "HAS_COMPLEXITY"
            for r in relations
        )
        if not already_has:
            for item in neighbors:
                text = (item.get("text") or "").strip()
                m = re.search(r"时间复杂度为\s*(O\([^)]+\)|O\(\w+\))", text)
                if m:
                    complexity_name = m.group(1)
                    # 确保复杂度实体存在
                    if complexity_name not in ent_by_name:
                        comp_ent = {"name": complexity_name, "type": "复杂度",
                                    "content_ids": [item["idx"]]}
                        entities.append(comp_ent)
                        ent_by_name[complexity_name] = comp_ent
                    relations.append({
                        "head": ent["name"],
                        "relation": "HAS_COMPLEXITY",
                        "tail": complexity_name,
                    })
                    break

    return entities, relations


# ── 合并去重 ──────────────────────────────────────────────────

def merge_across_batches(
    all_entities: list[dict],
    all_relations: list[dict],
    cache_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """三级合并: 精确去重 → LLM同义 → 关系重建"""
    # 第1级: 精确 name+type 合并
    merged = {}
    for e in all_entities:
        key = (e["name"], e["type"])
        if key not in merged:
            merged[key] = {"name": e["name"], "type": e["type"], "content_ids": []}
            for field in ("description", "pseudocode", "code", "aliases"):
                if e.get(field):
                    merged[key][field] = e[field]
        merged[key]["content_ids"].extend(e.get("content_ids", []))
        # 后出现的 code 可能更完整
        if e.get("code") and len(e.get("code", "")) > len(merged[key].get("code", "")):
            merged[key]["code"] = e["code"]
        if e.get("description") and len(e.get("description", "")) > len(merged[key].get("description", "")):
            merged[key]["description"] = e["description"]
    entities = list(merged.values())
    # 去重 content_ids
    for e in entities:
        e["content_ids"] = sorted(set(e["content_ids"]))
    print(f"   精确合并: {len(all_entities)} → {len(entities)}")

    # 第2级: LLM 同义合并
    entity_catalog = [{"name": e["name"], "type": e["type"]} for e in entities]
    merge_cache_path = None
    if cache_dir is not None:
        payload = json.dumps(
            {"model": LLM_MODEL, "system": MERGE_SYSTEM, "entities": entity_catalog},
            ensure_ascii=False,
            sort_keys=True,
        )
        merge_cache_path = cache_dir / (
            "merge-" + hashlib.sha256(payload.encode("utf-8")).hexdigest() + ".json"
        )
    if merge_cache_path is not None and merge_cache_path.exists():
        merge_rules = json.loads(merge_cache_path.read_text(encoding="utf-8"))["merges"]
        merge_cache_label = " [缓存]"
    else:
        resp = get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": MERGE_SYSTEM},
                {"role": "user", "content": json.dumps(entity_catalog, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=8192,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        merge_rules = json.loads(resp.choices[0].message.content.strip()).get("merges", [])
        merge_cache_label = ""
        if merge_cache_path is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp_path = merge_cache_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps({"merges": merge_rules}, ensure_ascii=False), encoding="utf-8"
            )
            temp_path.replace(merge_cache_path)
    # 代码侧硬约束：LLM 只能合并相同结构前缀下的操作节点。
    types_by_name = defaultdict(set)
    for entity in entities:
        types_by_name[entity["name"]].add(entity["type"])
    alias_map = {}
    accepted_rules = 0
    for rule in merge_rules:
        primary = rule.get("primary")
        if primary not in types_by_name or types_by_name[primary] != {"操作"}:
            continue
        primary_prefix, separator, _ = primary.partition("·")
        if not separator:
            continue
        accepted_alias = False
        for alias in rule.get("aliases", []):
            alias_prefix, alias_separator, _ = str(alias).partition("·")
            if (
                alias in types_by_name
                and types_by_name[alias] == {"操作"}
                and alias_separator
                and alias_prefix == primary_prefix
                and alias != primary
                and alias not in alias_map
            ):
                alias_map[alias] = primary
                accepted_alias = True
        accepted_rules += int(accepted_alias)
    print(
        f"   LLM 合并规则: 提议 {len(merge_rules)} 条，"
        f"代码约束接受 {accepted_rules} 条{merge_cache_label}"
    )

    final_entities = {}
    for e in entities:
        name = alias_map.get(e["name"], e["name"])
        if name not in final_entities:
            final_entities[name] = {**e, "name": name, "content_ids": list(e["content_ids"])}
            final_entities[name].setdefault("aliases", [])
        else:
            final_entities[name]["content_ids"].extend(e["content_ids"])
            for field in ("description", "pseudocode", "code"):
                if e.get(field) and len(e.get(field, "")) > len(final_entities[name].get(field, "")):
                    final_entities[name][field] = e[field]
    for e in final_entities.values():
        e["content_ids"] = sorted(set(e["content_ids"]))
    for alias, primary in alias_map.items():
        if primary in final_entities and alias not in final_entities[primary].get("aliases", []):
            final_entities[primary].setdefault("aliases", []).append(alias)

    entities = list(final_entities.values())
    print(f"   同义合并: → {len(entities)}")

    # 第3级: 关系去重 + 级联更新
    seen = set()
    clean_relations = []
    for r in all_relations:
        head = alias_map.get(r["head"], r["head"])
        tail = alias_map.get(r["tail"], r["tail"])
        key = (head, r["relation"], tail)
        if key not in seen:
            seen.add(key)
            clean_relations.append({"head": head, "relation": r["relation"], "tail": tail})
    print(f"   关系去重: {len(all_relations)} → {len(clean_relations)}")

    return entities, clean_relations


# ── Neo4j 导入 ────────────────────────────────────────────────

def import_to_neo4j(entities: list[dict], relations: list[dict]):
    if not NEO4J_PASSWORD:
        raise RuntimeError("未配置 NEO4J_PASSWORD，无法导入知识图谱")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def replace_graph(tx):
        # 只替换知识图谱标签，避免误删 Neo4j 中其他业务节点。
        tx.run(
            "MATCH (n) WHERE n:数据结构 OR n:操作 OR n:复杂度 "
            "DETACH DELETE n"
        )
        for entity in entities:
            label = entity["type"].replace(" ", "_")
            props = {
                key: value for key, value in entity.items()
                if key not in ("content_ids", "aliases") and value is not None
            }
            props["content_ids"] = json.dumps(entity.get("content_ids", []))
            props["aliases"] = json.dumps(entity.get("aliases", []))
            field_str = ", ".join(f"{key}: ${key}" for key in props)
            tx.run(f"CREATE (n:{label} {{{field_str}}})", **props)
        for relation in relations:
            query = (
                "MATCH (a {name: $head}) MATCH (b {name: $tail}) "
                f"CREATE (a)-[:{relation['relation']}]->(b)"
            )
            result = tx.run(query, head=relation["head"], tail=relation["tail"])
            result.consume()

    with driver.session() as session:
        # 单事务替换：任一节点或关系失败时由 Neo4j 自动回滚。
        session.execute_write(replace_graph)

    driver.close()
    print(f"✅ Neo4j 导入: {len(entities)} 实体, {len(relations)} 关系")


# ── 主流程 ──────────────────────────────────────────────────────

def load_global_contents(
    content_files: list[Path] | None = None,
) -> tuple[dict[Path, list[dict]], list[dict]]:
    """为每个 MinerU 元素分配稳定的“文档名 + 局部序号”来源 ID。"""
    contents_by_file = {}
    all_content = []
    for content_file in content_files or default_content_files():
        if not content_file.exists():
            continue
        content = json.loads(content_file.read_text(encoding="utf-8"))
        doc_name = content_file.stem.removesuffix("_content")
        for index, item in enumerate(content):
            item["idx"] = f"{doc_name}:content:{index:04d}"
        contents_by_file[content_file] = content
        all_content.extend(content)
    return contents_by_file, all_content


def load_existing_without_document(
    doc_name: str,
    target_content_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """移除目标文档的旧来源信息，同时保留其他文档及共享实体。"""
    entities_path = OUTPUT_DIR / "entities.json"
    relations_path = OUTPUT_DIR / "relations.json"
    if not entities_path.exists() or not relations_path.exists():
        return [], []

    old_entities = json.loads(entities_path.read_text(encoding="utf-8"))
    old_relations = json.loads(relations_path.read_text(encoding="utf-8"))
    retained_entities = []
    removed_names = set()
    source_prefix = f"{doc_name}_"

    for entity in old_entities:
        source_chunks = [
            chunk_id for chunk_id in entity.get("source_chunks", [])
            if not str(chunk_id).startswith(source_prefix)
        ]
        content_ids = [
            content_id for content_id in entity.get("content_ids", [])
            if content_id not in target_content_ids
        ]
        had_target_source = len(source_chunks) != len(entity.get("source_chunks", []))
        had_target_content = len(content_ids) != len(entity.get("content_ids", []))

        if (had_target_source or had_target_content) and not source_chunks and not content_ids:
            removed_names.add(entity["name"])
            continue

        cleaned = {key: value for key, value in entity.items() if key != "source_chunks"}
        cleaned["content_ids"] = content_ids
        retained_entities.append(cleaned)

    retained_relations = [
        relation for relation in old_relations
        if relation.get("head") not in removed_names and relation.get("tail") not in removed_names
    ]
    print(
        f"   既有图谱清理: 移除目标文档实体 {len(removed_names)} 个，"
        f"保留 {len(retained_entities)} 个实体 / {len(retained_relations)} 条关系"
    )
    return retained_entities, retained_relations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="只抽取指定文档，例如：数组")
    parser.add_argument(
        "--content-file",
        type=Path,
        action="append",
        help="显式指定 content.json；可重复传入，未指定时自动发现",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="成功抽取批次的内容哈希缓存目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="JSON 输出目录（默认 kb/graph）",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="定向抽取时，先移除该文档旧来源，再与现有图谱合并",
    )
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="只生成并校验 JSON，不写入 Neo4j",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="跳过抽取，直接把 output-dir 中已有 JSON 导入 Neo4j",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.import_only:
        entities = json.loads((args.output_dir / "entities.json").read_text(encoding="utf-8"))
        relations = json.loads((args.output_dir / "relations.json").read_text(encoding="utf-8"))
        import_to_neo4j(entities, relations)
        return

    print("=" * 60)
    print("Knowledge Graph 构建 v2 (Neo4j) — 多文档")
    print("=" * 60)

    content_files = [path.resolve() for path in args.content_file] if args.content_file else None
    contents_by_file, all_content = load_global_contents(content_files)
    selected_files = list(contents_by_file)
    if args.doc:
        selected_files = [path for path in selected_files if path.parent.name == args.doc]
        if not selected_files:
            raise FileNotFoundError(f"未找到文档的 content.json: {args.doc}")

    all_entities, all_relations = [], []
    extraction_errors = []
    if args.merge_existing:
        if not args.doc:
            raise ValueError("--merge-existing 必须和 --doc 一起使用")
        target_content_ids = {
            item["idx"] for content_file in selected_files
            for item in contents_by_file[content_file]
        }
        all_entities, all_relations = load_existing_without_document(
            args.doc, target_content_ids
        )

    for cf in selected_files:
        # 1. 加载（idx 已按所有标准 content.json 的固定顺序全局编号）
        content = contents_by_file[cf]
        items = [c for c in content if (c.get("text") or c.get("code_body") or "").strip()]
        print(f"\n📄 {cf.relative_to(BASE_DIR)}: {len(content)} 条, 有效 {len(items)}")

        # 2. 按章节分批
        chapters = split_by_chapter(items)
        print(f"📑 分 {len(chapters)} 个章节批次:")
        for title, batch in chapters:
            print(f"   [{len(batch)}条] {title[:70]}")

        # 3. 逐批提取
        print(f"\n🔄 调用 DeepSeek API 提取...")
        for ci, (chapter, batch) in enumerate(chapters, 1):
            print(f"   批次 {ci}/{len(chapters)} [{chapter[:50]}] ({len(batch)}条)...", end=" ", flush=True)
            try:
                result, cache_hit = extract_batch_cached(chapter, batch, args.cache_dir)
                all_entities.extend(result.get("entities", []))
                all_relations.extend(result.get("relations", []))
                cache_label = " [缓存]" if cache_hit else ""
                print(
                    f"实体 {len(result.get('entities',[]))}, "
                    f"关系 {len(result.get('relations',[]))}{cache_label}"
                )
            except Exception as e:
                print(f"失败: {e}")
                extraction_errors.append(f"{cf.name} / {chapter}: {e}")
                raise RuntimeError(
                    "当前图谱批次失败，已立即终止，禁止生成部分图谱。"
                ) from e

    if extraction_errors:
        raise RuntimeError(
            f"知识图谱抽取有 {len(extraction_errors)} 个批次失败，"
            "为防止部分结果污染正式图谱，已终止保存和导入。"
        )

    print(f"\n📊 原始抽取: {len(all_entities)} 实体, {len(all_relations)} 关系")

    # 4. 合并去重
    print(f"\n🔧 合并去重...")
    entities, relations = merge_across_batches(
        all_entities, all_relations, args.cache_dir
    )

    # 5. 全局补全（传入所有 content）
    print(f"\n🔧 全局补全 (code + HAS_COMPLEXITY)...")
    entities, relations = complete_missing(entities, relations, all_content)
    print(f"   补全后: {len(entities)} 实体, {len(relations)} 关系")

    # 6. 保存
    (args.output_dir / "entities.json").write_text(
        json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "relations.json").write_text(
        json.dumps(relations, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. 导入
    if args.no_import:
        print(f"\n⏸️  已跳过 Neo4j 导入，JSON 保存在 {args.output_dir}")
    else:
        print(f"\n📥 导入 Neo4j ({NEO4J_URI}) ...")
        import_to_neo4j(entities, relations)

    # 统计
    print(f"\n{'='*60}")
    print(f"🎉 完成")
    for t in ["数据结构", "操作", "复杂度"]:
        names = [e["name"] for e in entities if e["type"] == t]
        print(f"   {t} ({len(names)}): {', '.join(names[:8])}{'...' if len(names)>8 else ''}")
    for t in ["IS_A", "HAS_OPERATION", "HAS_COMPLEXITY"]:
        print(f"   {t}: {sum(1 for r in relations if r['relation']==t)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
