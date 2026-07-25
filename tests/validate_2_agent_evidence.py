"""
验证 2：Agent 证据评估能力
用 spec 的 P5 系统提示 + 模拟多轮场景，测 Agent 能否正确判断"够了就停，不够就继续搜"
"""
from app.config import create_llm_client, get_settings

# Spec 的 P5 系统提示（改编为单轮测试）
REACT_SYSTEM = """你是智慧教学系统的答疑 Agent。学生问你问题，你需要通过多轮检索找到充分证据，然后给出讲解。

【你可以调用的工具】（只有这 3 个）：
1. hybrid_retrieval(query) — 混合检索(BM25+向量+RRF+重排)，返回带 rerank_score 的 chunk 列表。
2. graph_lookup(concept) — 查知识图谱，返回该概念的 1 跳前置依赖（PREREQUISITE 边）。
3. fetch_by_kp(kp) — 按知识点名精确取出该知识点的核心讲解 chunk。

【你需要自己判断的事】（不通过调工具，在思考中完成）：
- 如果刚拿到的检索结果质量差（chunk 少或 rerank_score 低），在 Thought 里换一种问法，下一轮调 hybrid_retrieval 时传新 query。
- 如果判断证据缺乏前置知识，调 graph_lookup 看依赖链，结合学生画像判断哪些前置需要补。
- 拿到依赖后，自己判断要不要沿链继续追深——画像显示掌握度低→追，画像显示掌握度高→停。
- 证据够了 → 直接输出 Final Answer。注意难度适配学生画像（新手讲细、熟练讲深）。
- 如果 6 轮到了证据还不够 → 尽力答，但诚实标注"我不确定这部分说得对不对"。

【当前状态】
第 {iteration}/6 轮
学生问题：<question>{question}</question>
学生画像：<profile>{profile}</profile>
已收集的 chunk：
<evidence>
{evidence}
</evidence>

【请按 ReAct 格式输出】
Thought: 当前状态分析 + 下一步计划
Action: 调哪个工具 (hybrid_retrieval / graph_lookup / fetch_by_kp) 或 FINAL_ANSWER
如果 FINAL_ANSWER: 输出讲解
"""

# ── 测试场景 ──────────────────────────────────────────────────

# 场景 A: 证据充分 → 应该停在当前轮，输出 FINAL_ANSWER
SCENARIO_A = {
    "question": "顺序表插入操作的时间复杂度是多少？",
    "profile": "顺序表·插入:0.6, 时间复杂度:0.5",
    "evidence": """
[chunk_013 | rerank_score=0.89] 顺序表的插入操作是指在表的第 i-1 个元素和第 i 个元素之间插入一个新的元素 e。i 的取值范围为 0≤i≤n，当 i=n 时表示在末尾插入。
[chunk_015 | rerank_score=0.92] 在第 i 个元素之前插入时，需将第 n-1 至第 i 个元素（共 n-i 个）向后移动一个位置。时间复杂度为 O(n)。
[chunk_014 | rerank_score=0.78] 插入时需要检查表空间是否已满，检查插入位置有效性，注意数据移动方向为从后向前。
""",
    "iteration": 2,
    "expected": "应该选择 FINAL_ANSWER，因为证据已包含时间复杂度的答案和详细算法",
}

# 场景 B: 缺前置知识 → 应该调 graph_lookup 或 fetch_by_kp
SCENARIO_B = {
    "question": "B+树为什么比B树更适合做数据库索引？",
    "profile": "B+树:0.3, B树:0.2, 数据库索引:0.4",
    "evidence": """
[chunk_A | rerank_score=0.85] B+树的所有数据都存在叶子节点，内部节点只存索引键值。这使得范围查询非常高效。
[chunk_B | rerank_score=0.72] B+树的叶子节点通过指针相连形成有序链表，支持顺序遍历。
""",
    "iteration": 1,
    "expected": "应该调 graph_lookup('B+树') 查前置依赖，因为证据里没有 B 树的对比信息，且学生 B 树掌握度低",
}

# 场景 C: 检索结果差 → 应该换 query 重搜
SCENARIO_C = {
    "question": "顺序表删除操作的C代码怎么写？",
    "profile": "顺序表·删除:0.4, C语言:0.6",
    "evidence": """
[chunk_X | rerank_score=0.31] 线性表是由n个数据元素组成的有限序列，各元素之间存在线性关系。
[chunk_Y | rerank_score=0.28] 顺序存储结构用一组连续的存储单元依次存放数据元素。
""",
    "iteration": 1,
    "expected": "应该换 query 重搜（如 hybrid_retrieval('顺序表删除 C代码实现')），因为当前 chunk 完全不相关、分数极低",
}

# 场景 D: 第6轮仍不充分 → 应该降级输出
SCENARIO_D = {
    "question": "红黑树的删除修复共需要处理几种情况？请详细说明每种情况的调整过程。",
    "profile": "红黑树:0.1",
    "evidence": """
[chunk_RB1 | rerank_score=0.52] 红黑树是一种自平衡二叉搜索树，每个节点有红黑两种颜色。
[chunk_RB2 | rerank_score=0.41] 红黑树插入后有颜色调整和旋转操作。
""",
    "iteration": 6,
    "expected": "应该 FINAL_ANSWER 但诚实标注不确定性，因为已到 6 轮上限且证据不充分",
}

SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_D]


def run():
    client = create_llm_client()
    model = get_settings().llm_model
    print("=" * 70)
    print("验证 2：Agent 证据评估能力")
    print("=" * 70)

    for i, s in enumerate(SCENARIOS):
        print(f"\n{'─' * 70}")
        print(f"场景 {chr(65+i)}: {s['question'][:50]}...")
        print(f"画像: {s['profile']}")
        print(f"轮次: {s['iteration']}/6")
        print(f"期望行为: {s['expected']}")

        system = REACT_SYSTEM.format(
            iteration=s["iteration"],
            question=s["question"],
            profile=s["profile"],
            evidence=s["evidence"],
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "请输出你的 Thought 和 Action："},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"\n  LLM 输出:")
        for line in raw.split("\n"):
            print(f"  │ {line}")

        # 简单判定
        if "FINAL_ANSWER" in raw or "Final Answer" in raw:
            print(f"\n  🔵 判定: Agent 选择终止 → {'✅' if i in (0, 3) else '❌ 场景期望不停'} ")
        elif "graph_lookup" in raw:
            print(f"\n  🔵 判定: Agent 调 graph_lookup → {'✅' if i == 1 else '❌'} ")
        elif "hybrid_retrieval" in raw:
            print(f"\n  🔵 判定: Agent 调 hybrid_retrieval → {'✅' if i in (1, 2) else '❌'} ")
        elif "fetch_by_kp" in raw:
            print(f"\n  🔵 判定: Agent 调 fetch_by_kp → 待分析")
        else:
            print(f"\n  🔵 判定: 无法解析 Agent 意图")


if __name__ == "__main__":
    run()
