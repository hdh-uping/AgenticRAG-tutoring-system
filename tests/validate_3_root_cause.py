"""
验证 3：根因诊断推理
手工构造依赖链 + 学生画像，用 Prompt 9 测 LLM 能否沿依赖链追到真正薄弱根因
"""
import json

from app.config import create_llm_client, get_settings

# Spec 的 Prompt 9（改编为单轮测试，输入已是查询结果）
DIAGNOSIS_SYSTEM = """你是学习诊断专家。学生在某知识点上答错或卡住，请沿知识图谱的依赖链，定位真正的薄弱根因，而不是只看表面。

学生表现：<performance>{performance}</performance>

该知识点的依赖链（从Neo4j查询，表示"懂当前知识点需先懂什么"）：
<dependency_chain>
{dependency_chain}
</dependency_chain>

学生当前画像：
<profile>
{profile}
</profile>

诊断规则：
1. 不要只看学生卡住的表面知识点，要沿依赖链往下找
2. 找到"依赖链上掌握度<0.4"的节点，那才是真正根因
3. 例：学生AVL树错(0.2)，但AVL依赖BST，BST也才0.2 → 根因是BST没掌握，不是AVL
4. 给出置信度（0~1），表示你对这个根因判断有多确定

输出 JSON：
{{
  "diagnosis": [
    {{"weak_kp": "薄弱知识点", "confidence": 0.0~1.0, "reason": "为什么判断是它"}}
  ]
}}

约束：
- weak_kp 必须是依赖链上真实存在的知识点
- 沿依赖链追根因，不能只看表面
- 只输出 JSON"""

# ── 测试场景 ──────────────────────────────────────────────────

# 场景 A: 标准多跳场景——根因在深层
SCENARIO_A = {
    "label": "根因在深层（AVL→BST→二叉树）",
    "performance": "学生在 AVL 树的旋转操作上反复答错，表现出对平衡因子和旋转方向的理解混乱（掌握度 0.2）。",
    "dependency_chain": "(AVL树旋转)-[:前置]->(二叉搜索树)-[:前置]->(二叉树基础)-[:前置]->(线性表)",
    "profile": "AVL树旋转:0.2, 二叉搜索树:0.25, 二叉树基础:0.7, 线性表:0.8",
    "expected_root": "二叉搜索树",  # 第一个 <0.4 的节点，且本身也弱
}

# 场景 B: 根因就在表面
SCENARIO_B = {
    "label": "根因在表面（就是当前知识点没掌握）",
    "performance": "学生做顺序表插入操作时忘了检查插入位置合法性（掌握度 0.3）。",
    "dependency_chain": "(顺序表·插入)-[:前置]->(顺序表)-[:前置]->(线性表)",
    "profile": "顺序表·插入:0.3, 顺序表:0.75, 线性表:0.85",
    "expected_root": "顺序表·插入",  # 就是它自己，因为前置掌握度都高
}

# 场景 C: 多个薄弱点——选最深的
SCENARIO_C = {
    "label": "多个薄弱点，追最深根因（B+树→B树→磁盘IO→二叉树）",
    "performance": "学生无法理解 B+ 树为什么适合数据库索引，不知道内部节点和叶子节点的区别（掌握度 0.15）。",
    "dependency_chain": "(B+树)-[:前置]->(B树)-[:前置]->(磁盘IO局部性原理)-[:前置]->(二叉搜索树)",
    "profile": "B+树:0.15, B树:0.2, 磁盘IO局部性原理:0.3, 二叉搜索树:0.4",
    "expected_root": "磁盘IO局部性原理",  # 最深的薄弱点（<0.4），再往下 BST 0.4 刚好过线
}

# 场景 D: 依赖链不完整——边界情况
SCENARIO_D = {
    "label": "依赖链断层——只能追到数据有的地方",
    "performance": "学生不理解双向链表删除操作中的指针修改（掌握度 0.25）。",
    "dependency_chain": "(双向链表·删除)-[:前置]->(双向链表)-[:前置]->(单链表)",
    "profile": "双向链表·删除:0.25, 双向链表:0.35, 单链表:0.3",
    "expected_root": "单链表",  # 整条链都弱，追到最底层
}

SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_D]


def run():
    client = create_llm_client()
    model = get_settings().llm_model
    print("=" * 70)
    print("验证 3：根因诊断推理")
    print("=" * 70)

    for i, s in enumerate(SCENARIOS):
        print(f"\n{'─' * 70}")
        print(f"场景 {chr(65+i)}: {s['label']}")
        print(f"期望根因: {s['expected_root']}")

        system = DIAGNOSIS_SYSTEM.format(
            performance=s["performance"],
            dependency_chain=s["dependency_chain"],
            profile=s["profile"],
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "请输出诊断 JSON："},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  ❌ JSON 解析失败！")
            print(f"  原始: {raw[:300]}")
            continue

        diag = parsed.get("diagnosis", [])
        print(f"\n  诊断结果 ({len(diag)} 条):")
        for d in diag:
            kp = d.get("weak_kp", "?")
            conf = d.get("confidence", "?")
            reason = d.get("reason", "")[:120]
            match = "✅" if kp == s["expected_root"] else "⚠️"
            print(f"  {match} {kp} (置信度={conf})")
            print(f"     理由: {reason}...")

        # 分析
        top_kp = diag[0]["weak_kp"] if diag else "N/A"
        if top_kp == s["expected_root"]:
            print(f"\n  ✅ 根因判断正确: {top_kp}")
        elif any(d["weak_kp"] == s["expected_root"] for d in diag):
            print(f"\n  ⚠️  根因在列表中但不在首位: 首位={top_kp}, 期望={s['expected_root']}")
        else:
            print(f"\n  ❌ 根因判断偏差: 得到={top_kp}, 期望={s['expected_root']}")


if __name__ == "__main__":
    run()
