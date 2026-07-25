"""
验证 1：LLM 前置依赖抽取质量
用 Prompt 2 对 5 个典型 chunk 做知识点抽取，看输出稳不稳定、依赖合不合理
"""
import json

from app.config import create_llm_client, get_settings

# Prompt 2 from spec (unchanged)
SYSTEM_PROMPT = """你是《数据结构与算法》课程的知识图谱构建专家。

下面给你一段课程资料（用<chunk>标签包裹）。请从中抽取知识结构信息。

请抽取以下信息，严格按 JSON 格式输出：

1. source_kp：这段资料**最核心**讲的那个知识点（只填1个，名词短语）
2. knowledge_points：这段资料涉及的所有知识点（数组，包含 source_kp，2~5个）
3. prerequisites：这些知识点的前置依赖（即"要懂这些知识点，需要先懂什么"），每个前置依赖标注它依赖哪个知识点
4. difficulty：这段资料的难度（1=入门，2=基础，3=进阶，4=困难）

输出 JSON：
{
  "source_kp": "...",
  "knowledge_points": ["...", "..."],
  "prerequisites": [
    {"knowledge_point": "当前知识点", "depends_on": "前置知识点"},
    ...
  ],
  "difficulty": 3
}

示例（输入chunk讲"B+树内部节点只存索引"）：
{
  "source_kp": "B+树内部节点结构",
  "knowledge_points": ["B+树内部节点结构", "B+树", "B树"],
  "prerequisites": [
    {"knowledge_point": "B+树内部节点结构", "depends_on": "B树"},
    {"knowledge_point": "B树", "depends_on": "磁盘IO局部性原理"}
  ],
  "difficulty": 3
}

约束：
- source_kp 必须是名词短语，不能是句子
- prerequisites 的 depends_on 必须是真实存在的知识点概念，不能编造
- 如果某知识点没有前置依赖，prerequisites 中不写它
- 只输出 JSON，不要任何额外解释"""

# 5 个典型 chunk，覆盖不同难度
CHUNKS = [
    {
        "id": "chunk_003",
        "label": "线性表定义",
        "text": """线性表(Linear List)简称为表，是由 n(n≥0) 个数据元素(也叫节点或表元素)组成的有限序列。其特点是各数据元素之间存在着线性关系，即都是一个接一个地按一定顺序排列的，并且线性表要求同一个表中的各数据元素的结构类型必须完全一致。通常将线性表记作：(a₀, a₁, a₂, ..., aᵢ₋₁, aᵢ, aᵢ₊₁, ..., aₙ₋₁)。即表中 aᵢ₋₁ 领先于 aᵢ，称 aᵢ₋₁ 是 aᵢ 的直接前驱元素，aᵢ₊₁ 是 aᵢ 的直接后继元素。当 0≤i≤n-2 时，aᵢ 有且仅有一个直接后继；当 1≤i≤n-1 时，aᵢ 有且仅有一个直接前驱。线性表中元素的个数 n(n≥0) 定义为线性表的长度，特别地，当 n=0 时称该线性表为空表。""",
    },
    {
        "id": "chunk_006",
        "label": "顺序存储原理",
        "text": """线性表的顺序存储(Sequential Mapping，简称顺序表)，是指用一组地址连续的存储单元按线性表元素之间的逻辑顺序，依次存储线性表的数据元素。数据元素的逻辑顺序和物理上的存储顺序是完全一致的，物理上存放在位置 i 的元素，就是按照逻辑顺序存储时的第 i 个元素。因此在顺序存储结构下不需要另外建立空间来记录各个元素之间的关系。顺序存储的线性表是一种随机存取结构，因为只要确定了存储线性表的起始位置，就可以随机存取表中的任意一个数据元素。""",
    },
    {
        "id": "chunk_013",
        "label": "顺序表插入操作",
        "text": """顺序表的插入操作是指在表的第 i-1 个元素和第 i 个元素之间(即在第 i 个位置)插入一个新的元素 e，插入新元素后，表长为 n 的原表变为表长为 n+1 的新表。其中需要注意的是 i 的取值范围为 0≤i≤n，当 i=n 时，表示在整个顺序表的末尾插入一个元素 e。""",
    },
    {
        "id": "chunk_from_later_chapter",
        "label": "单链表节点结构",
        "text": """单链表(Linked List)是一种链式存储结构。在单链表中，每个数据元素被存储在一个节点(Node)中。每个节点包含两个域：数据域(Data Field)用于存放数据元素本身的信息；指针域(Next 或 Link Field)用于存放指向该节点直接后继节点的指针。单链表的节点结构如下图所示。由于单链表的每个节点只有一个指针域，因此称为单链表。单链表通过每个节点的指针域将 n 个节点按线性表的逻辑顺序链接在一起。""",
    },
    {
        "id": "chunk_tricky",
        "label": "线性表特点（多条规则）",
        "text": """通过定义可以得到线性表有如下特点:(1) 唯一首元素;(2) 唯一尾元素;(3) 除首元素外，任何元素都有一个前驱；(4) 除尾元素外，任何元素都有一个后继；(5) 每个元素有一个位序。在稍复杂的线性表中，一个数据元素可以由若干数据项(Item)组成，此时这种数据元素称为记录(Record)。含有大量记录的线性表又称为文件(File)。""",
    },
]


def run():
    client = create_llm_client()
    model = get_settings().llm_model
    results = []
    for ch in CHUNKS:
        user = f"<chunk>\n{ch['text']}\n</chunk>"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        # 去掉可能的 markdown 包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            parsed = json.loads(raw)
            results.append({"id": ch["id"], "label": ch["label"], "parsed": parsed, "raw": raw})
        except json.JSONDecodeError:
            results.append({"id": ch["id"], "label": ch["label"], "parse_error": True, "raw": raw})

    # 输出
    print("=" * 70)
    print("验证 1：LLM 前置依赖抽取质量")
    print("=" * 70)

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"[{r['id']}] {r['label']}")
        if r.get("parse_error"):
            print(f"  ❌ JSON 解析失败！原始输出:")
            print(f"  {r['raw'][:300]}")
            continue

        p = r["parsed"]
        print(f"  source_kp:       {p.get('source_kp', 'MISSING')}")
        print(f"  knowledge_points: {p.get('knowledge_points', 'MISSING')}")
        print(f"  difficulty:       {p.get('difficulty', 'MISSING')}")

        prereqs = p.get("prerequisites", [])
        if prereqs:
            print(f"  prerequisites ({len(prereqs)}):")
            for pr in prereqs:
                kp = pr.get("knowledge_point", "?")
                dep = pr.get("depends_on", "?")
                print(f"    {kp}  ←依赖←  {dep}")
        else:
            print(f"  prerequisites:    (无)")

        # 合理性检查
        issues = []
        source = p.get("source_kp", "")
        if not source or len(source) < 2:
            issues.append("source_kp 太短或缺失")
        if "。" in source or "，" in source:
            issues.append("source_kp 含标点，不是名词短语")
        kps = p.get("knowledge_points", [])
        if source and source not in kps:
            issues.append("source_kp 不在 knowledge_points 中")
        for pr in prereqs:
            dep = pr.get("depends_on", "")
            if dep == "" or len(dep) < 2:
                issues.append(f"depends_on 可疑: '{dep}'")
            if "要" in dep or "需要" in dep or "理解" in dep:
                issues.append(f"depends_on 像解释不是概念名: '{dep}'")

        if issues:
            for iss in issues:
                print(f"  ⚠️  {iss}")
        else:
            print(f"  ✅ 格式检查通过")


if __name__ == "__main__":
    run()
