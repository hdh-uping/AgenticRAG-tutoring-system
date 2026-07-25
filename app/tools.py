"""
Agent 工具层 —— 封装 hybrid_retrieval 和 graph_lookup 两个工具。
加载一次全局复用，返回格式化文本供 Agent 阅读。
"""
import json
import logging
from pathlib import Path

from app.config import ConfigurationError, get_settings
from app.local_graph import lookup as local_graph_lookup

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"
SETTINGS = get_settings()
logger = logging.getLogger(__name__)

CHUNKS_DIR = KB_DIR / "data"
VECTORDB_FILE = str(SETTINGS.milvus_db_path)
COLLECTION_NAME = SETTINGS.milvus_collection
EMBED_MODEL_PATH = str(SETTINGS.embed_model_path)
RERANK_MODEL_PATH = str(SETTINGS.rerank_model_path)

# ── 全局状态（首次调用时加载） ────────────────────────────────
_embed_model = None
_milvus_client = None
_bm25_retriever = None
_graph_driver = None
_all_chunks = None
_reranker_model = None
_reranker_tokenizer = None


def _load_embed():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        if not Path(EMBED_MODEL_PATH).exists():
            raise ConfigurationError(f"Embedding 模型目录不存在: {EMBED_MODEL_PATH}")
        _embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")
    return _embed_model


def _load_milvus():
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient

        if not Path(VECTORDB_FILE).exists():
            raise ConfigurationError(f"Milvus Lite 数据库不存在: {VECTORDB_FILE}")
        _milvus_client = MilvusClient(VECTORDB_FILE)
        _milvus_client.load_collection(COLLECTION_NAME)
    return _milvus_client


def _load_chunks():
    global _all_chunks
    if _all_chunks is None:
        _all_chunks = []
        for fpath in sorted(CHUNKS_DIR.glob("chunks_*.jsonl")):
            with open(fpath, encoding="utf-8") as f:
                _all_chunks.extend(json.loads(l) for l in f if l.strip())
    return _all_chunks


def _load_bm25():
    global _bm25_retriever
    if _bm25_retriever is None:
        chunks = _load_chunks()
        _bm25_retriever = BM25Retriever(chunks)
    return _bm25_retriever


def _load_reranker():
    global _reranker_tokenizer, _reranker_model
    if _reranker_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(RERANK_MODEL_PATH).exists():
            raise ConfigurationError(f"Reranker 模型目录不存在: {RERANK_MODEL_PATH}")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            RERANK_MODEL_PATH, padding_side="left"
        )
        if _reranker_tokenizer.pad_token_id is None:
            _reranker_tokenizer.pad_token = _reranker_tokenizer.eos_token
        _reranker_model = AutoModelForCausalLM.from_pretrained(
            # torch_dtype 兼容项目允许的 Transformers 4.51；新版本只会给弃用警告。
            RERANK_MODEL_PATH, torch_dtype=torch.float16, device_map="cpu"
        )
        _reranker_model.eval()
    return _reranker_tokenizer, _reranker_model


def _load_graph():
    global _graph_driver
    if _graph_driver is None:
        from neo4j import GraphDatabase

        password = SETTINGS.require_neo4j_password()
        _graph_driver = GraphDatabase.driver(
            SETTINGS.neo4j_uri,
            auth=(SETTINGS.neo4j_user, password),
        )
    return _graph_driver


# ── BM25 ──────────────────────────────────────────────────────

class BM25Retriever:
    STOP_WORDS = set(
        "的 是 怎么 一个 什么 这个 那个 可以 我们 进行 需要 其中 没有 "
        "对于 以及 使用 不能 不会 能够 不是 这种 那种 因为 所以 但是 "
        "如果 虽然 因此 而且 或者 已经 比较 非常 为主 表示 包括 例如 "
        "即 等 在 了 有 和 就 不 也 这 都 要 到 将 为 与 对 从 被 把 着"
        .split()
    )

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in self.jieba.cut(text) if w.strip() and w not in self.STOP_WORDS]

    def __init__(self, chunks: list[dict]):
        import jieba
        from rank_bm25 import BM25Okapi

        self.jieba = jieba
        for term in ["顺序表", "单链表", "循环链表", "双向链表", "线性表",
                     "时间复杂度", "头指针", "头节点", "数据元素", "直接前驱", "直接后继",
                     "头插入法", "尾插入法", "按值查找", "取元素"]:
            self.jieba.add_word(term)
        self.corpus = [self._tokenize(c["text"]) for c in chunks]
        self.chunks = chunks
        self.bm25 = BM25Okapi(self.corpus, k1=1.2, b=0.75)

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        raw = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(raw), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = round(float(score), 5)
            results.append((chunk, float(score)))
        return results


# ── Reranker ──────────────────────────────────────────────────

def _binary_relevance_scores(logits, no_id: int, yes_id: int):
    """按 Qwen3-Reranker 定义，用 no/yes 两个 logit 计算相关概率。"""
    import torch

    binary_logits = torch.stack(
        [logits[:, no_id], logits[:, yes_id]], dim=1
    ).float()
    return torch.softmax(binary_logits, dim=1)[:, 1]


def _rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    import torch

    tokenizer, model = _load_reranker()
    no_id = tokenizer.convert_tokens_to_ids("no")
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    task = "Given a web search query, retrieve relevant passages that answer the query"
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    max_length = 2048

    pairs = [
        f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {candidate['text']}"
        for candidate in candidates
    ]
    tokenized = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    input_ids = [
        prefix_tokens + item + suffix_tokens
        for item in tokenized["input_ids"]
    ]
    inputs = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        scores = _binary_relevance_scores(logits, no_id, yes_id).cpu().tolist()

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 5)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# ── RRF 融合 ──────────────────────────────────────────────────

def _rrf_fusion(vec_results: list, bm25_results: list, k: int = 60) -> list[dict]:
    rrf_scores = {}
    for result_list in [vec_results, bm25_results]:
        for rank, (chunk, _) in enumerate(result_list, start=1):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

    chunk_map = {}
    for rl in [vec_results, bm25_results]:
        for chunk, _ in rl:
            chunk_map.setdefault(chunk["id"], chunk)

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [dict(chunk_map[cid]) for cid in sorted_ids]


# ═══════════════════════════════════════════════════════════════
# 工具 1：hybrid_retrieval
# ═══════════════════════════════════════════════════════════════

def hybrid_retrieval(query: str, top_k: int = 5) -> str:
    """混合检索：BM25 + 向量 → RRF → Reranker → 返回带分数的 chunk 清单。"""
    return format_retrieval_results(hybrid_retrieval_results(query, top_k=top_k))


def hybrid_retrieval_results(query: str, top_k: int = 5) -> list[dict]:
    """执行混合检索并返回结构化证据，便于评测和其他调用方复用。"""
    embed = _load_embed()
    milvus = _load_milvus()
    bm25 = _load_bm25()

    # 向量检索
    vec = embed.encode(
        ["Instruct: Given a web search query, retrieve relevant passages "
         f"that answer the query\nQuery: {query}"],
        normalize_embeddings=True,
    )[0].tolist()
    hits = milvus.search(
        collection_name=COLLECTION_NAME,
        data=[vec],
        output_fields=["id", "text", "header_path", "page_num", "char_count"],
        limit=10,
    )[0]
    vec_results = []
    for h in hits:
        c = dict(h["entity"])
        c["vector_score"] = round(h["distance"], 5)
        vec_results.append((c, h["distance"]))

    # BM25 检索
    bm25_results = bm25.search(query, top_k=10)

    # RRF 融合
    fused = _rrf_fusion(vec_results, bm25_results)

    # Reranker 精排
    final = _rerank(query, fused, top_k)

    return final


def format_retrieval_results(results: list[dict]) -> str:
    """将结构化证据格式化为 Agent 可读文本。"""
    if not results:
        return "(未找到相关内容)"

    parts = []
    for i, c in enumerate(results, 1):
        score = c.get("rerank_score", 0)
        page = c.get("page_num", "?")
        header = c.get("header_path", "")[:80]
        text = c["text"][:400].replace("\n", " ")
        parts.append(
            f"[Chunk{i} | id={c.get('id', '?')} | rerank_score={score:.4f} "
            f"| 第{page}页 | {header}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# 工具 2：graph_lookup
# ═══════════════════════════════════════════════════════════════

def graph_lookup(concept: str) -> str:
    """查 Neo4j 知识图谱，返回指定概念的结构化信息。"""
    if not SETTINGS.neo4j_password:
        return local_graph_lookup(concept)

    try:
        return _graph_lookup_neo4j(concept)
    except Exception:
        logger.exception("Neo4j 查询失败，降级为本地 JSON 图谱")
        return local_graph_lookup(concept)


def _graph_lookup_neo4j(concept: str) -> str:
    driver = _load_graph()

    with driver.session() as session:
        # 1. 构造多个搜索变体（处理 · 分隔符和空格问题）
        keywords = [concept]  # 原始输入
        if "·" in concept:
            keywords.append(concept.replace("·", ""))  # "顺序表·插入" → "顺序表插入"
        else:
            # "顺序表插入" → 尝试 "顺序表·插入"
            for known in ["顺序表", "单链表", "循环链表", "双向链表", "线性表"]:
                if known in concept:
                    rest = concept.replace(known, "")
                    keywords.append(f"{known}·{rest}")
                    break

        # 2. 先精确匹配，再做有序模糊匹配，避免 CONTAINS + LIMIT 1 随机命中。
        record = None
        for kw in keywords:
            result = session.run(
                "MATCH (n {name: $kw}) RETURN n, labels(n) AS labels LIMIT 1",
                kw=kw,
            )
            records = list(result)
            if records:
                record = records[0]
                break

        if not record:
            for kw in keywords:
                result = session.run(
                    "MATCH (n) WHERE n.name CONTAINS $kw "
                    "RETURN n, labels(n) AS labels "
                    "ORDER BY size(n.name), n.name LIMIT 1",
                    kw=kw,
                )
                records = list(result)
                if records:
                    record = records[0]
                    break

        if not record:
            for kw in keywords:
                result = session.run(
                    "MATCH (n) WHERE $kw CONTAINS n.name "
                    "RETURN n, labels(n) AS labels "
                    "ORDER BY size(n.name) DESC, n.name LIMIT 1",
                    kw=kw,
                )
                records = list(result)
                if records:
                    record = records[0]
                    break

        if not record:
            return f"(图谱中未找到与「{concept}」相关的节点。已尝试关键词: {', '.join(keywords)})"

        node = dict(record["n"])
        labels = record["labels"]
        name = node.get("name", concept)
        label = labels[0] if labels else "未知"

        parts = [f"[{label}] {name}"]

        has_full_info = bool(node.get("code"))
        if has_full_info:
            parts.append("  ℹ️  以上已包含完整代码和步骤，可直接用于回答。")

        for field, field_cn in [("description", "描述"), ("pseudocode", "步骤"), ("code", "代码")]:
            if node.get(field):
                val = node[field]
                if len(val) > 600:
                    val = val[:600] + "..."
                parts.append(f"  {field_cn}: {val}")

        # 3. 查关联操作（如果当前是数据结构）
        if label == "数据结构":
            ops = session.run(
                "MATCH (a {name: $name})-[:HAS_OPERATION]->(op) RETURN op.name",
                name=name,
            ).values()
            if ops:
                parts.append(f"  操作列表: {', '.join(r[0] for r in ops)}")

            # IS_A 子类型
            subs = session.run(
                "MATCH (sub)-[:IS_A]->(a {name: $name}) RETURN sub.name",
                name=name,
            ).values()
            if subs:
                parts.append(f"  子类型: {', '.join(r[0] for r in subs)}")

            # IS_A 父类型
            parents = session.run(
                "MATCH (a {name: $name})-[:IS_A]->(p) RETURN p.name",
                name=name,
            ).values()
            if parents:
                parts.append(f"  属于: {', '.join(r[0] for r in parents)}")

        # 4. 查复杂度（如果当前是操作）
        if label == "操作":
            comps = session.run(
                "MATCH (op {name: $name})-[:HAS_COMPLEXITY]->(c) RETURN c.name",
                name=name,
            ).values()
            if comps:
                parts.append(f"  时间复杂度: {', '.join(r[0] for r in comps)}")

    return "\n".join(parts)
