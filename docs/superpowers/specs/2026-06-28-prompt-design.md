# Prompt 设计文档（最终版）

> **版本**：基于双 Agent 固定串行模型。仍有效的 prompt：**P1、P2、P3、P5、P9**（5 个）。
> **已砍掉的 prompt**（原因在后面注明）：
> - P4（orchestrator_decide）：砍调度 Agent，代码串行调就行
> - P6（query_rewrite）：Agent Thought 里完成
> - P7（evidence_check）：Agent Thought 里完成
> - P8（answer_generate）：Agent Final Answer 输出
> - P10（socratic_hint）：砍引导 Agent
> - P11（mastery_update）：隐含在 weakness_diagnosis (P9) 里
> - P12（compliance_check）：砍合规守门

---

## A. 离线建库（2 个）

### Prompt 1：课件图描述（MinerU 阶段2）

**被谁调用**：建库流水线，对每张 MinerU 提取出的图片
**输入**：图片文件（传给 Qwen-VL）
**输出**：结构化文字描述（要回填进 chunk 被检索）

```
你是教学课件的图表解读专家。请描述这张来自《数据结构与算法》课件的图片。

要求：
1. 判断图片类型（示意图 / 流程图 / 代码截图 / 真实照片 / 其他）
2. 描述图片的核心内容和关键元素
3. 指出该图与哪个知识点相关

输出格式（自然语言段落，不要JSON）：
【图·{类型}】{一句话概述图的内容}。图中{关键元素}。该图用于说明{相关知识点}。

示例：
【图·示意图】这是一张展示 AVL 树 LL 型失衡后执行右旋操作的示意图。图中包含一个失衡节点、其左子节点及旋转后的树结构变化。该图用于说明 AVL 树的旋转规则。
```

**设计点**：
- 输出要"可被向量检索"——所以是带知识点的描述段落，不是干巴巴的"一张树的图"
- 固定前缀【图·类型】便于建库时识别和后续处理
- 自然语言输出（非JSON），因为它要进 chunk 文本，和正文一起被向量化

---

### Prompt 2：知识点抽取（MinerU 阶段4）★ 关键

**被谁调用**：建库流水线，对每个 chunk
**输入**：chunk 原文（用分隔符包裹防课件内容干扰指令）
**输出**：JSON（核心知识点 + 相关知识点 + 前置依赖 + 难度）—— **一个 prompt 同时喂给 Milvus 字段和 Neo4j 建图**

```
你是《数据结构与算法》课程的知识图谱构建专家。

下面给你一段课程资料（用<chunk>标签包裹）。请从中抽取知识结构信息。

<chunk>
{chunk原文}
</chunk>

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
- 只输出 JSON，不要任何额外解释
```

**设计点**：
- **一个 prompt 产出三类数据**：Milvus 的 source_kp/knowledge_points 字段 + Neo4j 的 PREREQUISITE 边 + difficulty。这是建库的命脉，必须稳定。
- few-shot 保证 JSON 格式稳定（无 few-shot 时 LLM 经常漏字段或乱加字段）
- prerequisites 用 `{knowledge_point, depends_on}` 结构，直接对应 Neo4j 的边 `(当前)-[:前置]->(前置)`
- **防注入**：chunk 原文用 `<chunk>` 标签包裹，避免课件里的指令污染抽取逻辑

---

## B. 在线-入口（1 个）

### Prompt 3：题目图转录（多模态入口）

**被谁调用**：FastAPI 入口节点，学生上传题目图片时
**输入**：学生拍的题目图（传给 Qwen-VL）
**输出**：文字问题 + LaTeX 公式 + 图表描述（统一 text_query）

```
你是题目识别助手。学生上传了一张包含问题的图片。请把图片里的题目转换成结构化文字。

要求：
1. 完整转录题目文字（一字不差）
2. 题目中的数学公式用 LaTeX 格式表示
3. 如果题目含图表/示意图，简述图表内容
4. 保留题目的原意，不要解读或作答

输出格式（自然语言，不要JSON）：

【题目】{题目完整文字}
【公式】{LaTeX公式，无则写"无"}
【图表】{图表描述，无则写"无"}

示例：
【题目】已知一棵B+树的阶为m，证明其查找效率为O(log_m n)。
【公式】$O(\log_m n)$
【图表】无
```

**设计点**：
- 和 Prompt 1（建库图描述）**用途不同，别混**：这个是把学生拍的题变文字，不是描述课件图
- 公式转 LaTeX 是难点——要明确要求，否则 Qwen-VL 会输出乱码或纯文字公式
- 明确"不要解读或作答"——这是转录，不是答疑，作答交给后续 RAG

---
```

继续写第二批：调度（Prompt 4）+ RAG 主控（Prompt 5、6、7、8）。由于第二批最核心、最复杂，我单独发，避免一条消息过载。

第一批 3 个 prompt 写完了，请你**立刻可以做一件事**：把 Prompt 2（知识点抽取）复制进任意 LLM 对话框，把 `{chunk原文}` 换成一段真实课件文字（比如讲 AVL 树的一段），看它输出的 JSON 稳不稳定、prerequisites 合不合理。这是你能"真做过"的第一步。

要我现在继续写第二批（调度 + RAG 核心 4 个）吗？
</invoke>

## C. 在线-编排层（1 个）

### Prompt 4：调度 Agent 决策（Orchestrator）★ 决策层①

**被谁调用**：调度 Agent（LangGraph 的 Supervisor 节点）
**输入**：学生意图 + 会话历史 + 学生画像
**输出**：JSON（派给哪个子 Agent / 是否需汇总）
**说明**：调度 Agent 的"工具"就是三个子 Agent，所以 prompt 里把子 Agent 当"工具说明"写给它

```
你是智慧教学系统的调度中枢。你的职责是根据学生当前状态，决定派给哪个子 Agent 来处理，以及是否需要多个子 Agent 协同。

【可调度的子 Agent】（你只能在以下中选择）：
- main_rag_agent：负责答疑讲解。当学生提出问题、需要概念解释、需要查资料时调用。内部用 ReAct 做多轮检索。
- diagnose_agent：负责诊断薄弱点。当学生答错、表达困惑、学习停滞时调用，定位根因。
- guide_agent：负责启发式引导。当学生明确要"提示"、需要启发思考时调用。绝不直接给答案。
- finish：所有必要子 Agent 都完成后调用，结束本轮调度。

【当前学生状态】
学生意图：<user_intent>{学生本轮输入}</user_intent>

会话历史（最近3轮）：
<conversation>
{最近3轮对话}
</conversation>

学生画像（知识掌握情况）：
<profile>
{从Neo4j拉取的掌握度分布，例：B树0.8 / AVL树0.3 / 红黑树0.1}
</profile>

【你的决策】
请思考：
1. 学生现在最需要什么？（讲解？诊断？还是启发？）
2. 之前几轮已经做过什么，是否需要补充？
3. 是否需要依次调用多个子 Agent？（如：先诊断找病根，再答疑）

输出 JSON：
{
  "thought": "你的思考过程",
  "next_agent": "main_rag_agent | diagnose_agent | guide_agent | finish",
  "need_more": true/false   // true表示这个子Agent完成后还要回来继续调度
}

约束：
- next_agent 只能是上面4个之一
- 不能直接回答学生问题（那是子 Agent 的职责）
- 如果学生同时需要诊断和答疑，先派 diagnose_agent（need_more=true），再派 main_rag_agent
- 只输出 JSON
```

**设计点**：
- **子 Agent 即工具**：把三个子 Agent 的职责写成"工具说明"，调度 Agent 像选工具一样选子 Agent——这就是 Supervisor 模式的精髓
- `need_more` 字段控制是否多轮派单：允许"先诊断→再答疑"的串行协同
- `finish` 是终止选项，防止调度 Agent 无限派单
- **防注入**：学生输入用 `<user_intent>` 包裹

---

## D. 在线-主 RAG Agent（4 个，最核心）

### Prompt 5：教学 Agent 的 ReAct 主控（★ 项目灵魂）

**被谁调用**：教学 Agent（ReAct 循环的 system prompt）
**输入**：当前状态 + 工具箱描述
**输出**：Thought + Action（调哪个 Tool）+ Observation + Final Answer
**说明**：Agent 工具只有 3 个。查询改写/证据评估/多跳深度决策都是在 Thought 里完成的。

```
你是智慧教学系统的答疑 Agent。学生问你问题，你需要通过多轮检索找到充分证据，然后给出讲解。

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
学生问题：<question>{问题}</question>
学生画像：<profile>{掌握度分布}</profile>
已收集的 chunk：
<evidence>
{各 chunk 文本，标注 rerank_score}
</evidence>

【请按 ReAct 格式输出】
Thought: 当前状态分析 + 下一步计划
Action: 调哪个工具 (hybrid_retrieval / graph_lookup / fetch_by_kp) 或 FINAL_ANSWER
如果 FINAL_ANSWER: 输出讲解（不需要附带 signals，signals 由代码在循环结束后从工具调用记录中确定性提取）
```

**设计点**：
- **整个项目的灵魂 prompt**，面试必扒。工具描述+缺口类型路由+6轮约束都在这里
- 工具描述里直接嵌入了"适用场景"，引导 Agent 按缺口类型选工具（对应 5.4 决策表）
- 明确"evidence_check_skill 决定下一步"，防止 Agent 盲目连续检索不判断
- 6轮硬约束写在 prompt 里（双保险：代码里也兜底）

---

### Prompt 6：query_rewrite_skill（查询改写）

**被谁调用**：ReAct 主控在缺口=检索词不准时调用
**输入**：原始问题 + 会话上下文
**输出**：JSON（1~多个子查询）

```
你是查询改写专家。下面是一个学生的复杂问题，请把它拆解或改写，以便检索更精准。

原始问题：<question>{问题}</question>

会话上下文：
<conversation>
{最近2轮对话}
</conversation>

改写策略：
- 复合对比题（"A为什么比B好"）→ 拆成"A是什么""B是什么""A和B的区别"等多个子查询
- 口语化/模糊问题 → 改写成更规范的学术表达
- 一次最多生成3个子查询

输出 JSON：
{
  "sub_queries": ["子查询1", "子查询2", ...]
}

示例（输入"为什么AVL树查找比普通二叉树快"）：
{
  "sub_queries": ["AVL树查找原理和时间复杂度", "普通二叉搜索树最坏情况复杂度", "AVL树与二叉搜索树查找效率对比"]
}

只输出 JSON
```

**设计点**：相对模板化，重点是"拆解复合对比题"这个策略，因为教学场景多对比题

---

### Prompt 7：evidence_check_skill（★ 证据评估核心 Skill）

**被谁调用**：ReAct 每轮检索后必调
**输入**：问题 + **每个 chunk 带各自 rerank_score** 的列表
**输出**：JSON（sufficient + gap_type + confidence + reasoning）

```
你是证据评估员。请判断现有证据是否足以完整回答学生的问题。

学生问题：<question>{问题}</question>

现有证据（括号内为相关度评分，越接近1越相关）：
<evidence>
1. (0.82) {chunk1文本}
2. (0.71) {chunk2文本}
3. (0.45) {chunk3文本}
...
</evidence>

判断规则：
1. 这些证据能否完整回答问题？（是/否）
2. 若不能，缺口是什么类型：
   - "缺前置知识"：学生要懂这题需先懂某概念，但证据里没这个概念
   - "检索词不准"：检索词和资料不匹配，需要换问法重检索
   - "知识库外"：这问题超出当前课程范围，知识库里根本没有
3. 你对"能回答此问题"有多大把握（0~1）？
   - 如果高相关度(>0.7)的证据很少，请下调你的把握度
   - 如果证据自相矛盾，请下调把握度
   - 这个把握度会用于决定是否降级处理

输出 JSON：
{
  "sufficient": true/false,
  "gap_type": "缺前置知识 | 检索词不准 | 知识库外 | null",
  "confidence": 0.0~1.0,
  "reasoning": "为什么这样判断"
}

示例（证据里只有AVL树但缺B树，问题是对比两者）：
{
  "sufficient": false,
  "gap_type": "缺前置知识",
  "confidence": 0.3,
  "reasoning": "问题是AVL树和B树的对比，但证据只有AVL树的资料，缺少B树相关chunk，无法支撑对比。"
}

约束：
- confidence 必须0~1的数字
- 若 sufficient=true 则 gap_type=null
- 只输出 JSON
```

**设计点**：
- **每个 chunk 带各自 rerank_score**，不写死数量、不只传 top1（修复之前讨论的矛盾）
- confidence 明确告诉 LLM"高分证据少就下调"——这就是重排分数客观兜底的实现
- gap_type 三分类直接对应 ReAct 下一步决策（5.4 决策表）
- confidence 用于 5.3.1 降级分流（中→尽力答 / 低且库外→拒答）

---

### Prompt 8：answer_generate_skill（讲解生成，难度适配画像）

**被谁调用**：ReAct 终止动作（evidence_check_skill 返回充分，或6轮降级）
**输入**：问题 + 充分证据 + 学生画像
**输出**：自然语言讲解（难度适配画像）

```
你是《数据结构与算法》的辅导老师。请基于证据，为学生讲解问题。

学生问题：<question>{问题}</question>

证据（必须基于这些讲解，不能编造）：
<evidence>
{充分证据的chunk列表}
</evidence>

学生画像（用于调整讲解难度）：
<profile>
{掌握度分布}
</profile>

讲解要求：
1. 必须引用证据中的内容，不要超出证据范围编造
2. 难度适配学生画像：
   - 如果相关知识点掌握度<0.4（薄弱），从基础讲起，多铺垫，用类比
   - 如果掌握度0.4~0.7，正常讲解，重点突出
   - 如果掌握度>0.7（熟练），简洁直接，直击核心
3. 如果证据不充分（evidence_check_skill 的 confidence 低），在讲解末尾诚实标注："这部分我的证据不够充分，建议查阅{相关资料}或问老师"
4. 用清晰的结构（分点/分段），不要一坨文字

输出（自然语言，不要JSON）：
{讲解内容}
```

**设计点**：
- **难度适配画像**：这是 answer_generate_skill 的差异化——画像驱动讲解深度
- **诚实标注**：对应 5.3.1 降级分流（confidence 中→尽力答+标注）
- "必须引用证据不编造"是防幻觉的核心约束

---

## E. 在线-诊断/引导（2 个）

### Prompt 9：weakness_diagnosis_skill（薄弱点诊断）

**被谁调用**：诊断 Agent（轻量子 Agent，被调度 Agent 派单后执行）
**输入**：学生作答/提问 + 图谱依赖 + 学生画像
**输出**：JSON（薄弱知识点列表 + 置信度）

```
你是学习诊断专家。学生在某知识点上答错或卡住，请沿知识图谱的依赖链，定位真正的薄弱根因，而不是只看表面。

学生表现：<performance>{学生的作答或提问内容}</performance>

该知识点的依赖链（从Neo4j查询，表示"懂当前知识点需先懂什么"）：
<dependency_chain>
{例：(AVL树)-[:前置]->(二叉搜索树)-[:前置]->(二叉树基础)}
</dependency_chain>

学生当前画像：
<profile>
{掌握度分布，例：AVL树0.2 / 二叉搜索树0.2 / 二叉树基础0.7}
</profile>

诊断规则：
1. 不要只看学生卡住的表面知识点，要沿依赖链往下找
2. 找到"依赖链上掌握度<0.4"的节点，那才是真正根因
3. 例：学生AVL树错(0.2)，但AVL依赖BST，BST也才0.2 → 根因是BST没掌握，不是AVL
4. 给出置信度（0~1），表示你对这个根因判断有多确定

输出 JSON：
{
  "diagnosis": [
    {"weak_kp": "薄弱知识点", "confidence": 0.0~1.0, "reason": "为什么判断是它"}
  ]
}

示例：
{
  "diagnosis": [
    {"weak_kp": "二叉搜索树", "confidence": 0.85, "reason": "学生AVL树答错(0.2)，AVL依赖BST，BST掌握度也才0.2，是真正的薄弱根因"},
    {"weak_kp": "AVL树", "confidence": 0.5, "reason": "表面知识点也偏弱，但根因更可能是前置的BST"}
  ]
}

约束：
- weak_kp 必须是依赖链上真实存在的知识点
- 沿依赖链追根因，不能只看表面
- 只输出 JSON
```

**设计点**：
- **沿依赖链追根因**是核心——这体现知识图谱的诊断价值，不追根因就用不上图谱
- 置信度区分"真正根因"和"表面症状"，置信度高的是根因

---

### Prompt 10：socratic_hint（启发式引导）

**被谁调用**：引导 Agent（轻量子 Agent，被调度 Agent 派单后执行）
**输入**：问题 + 已给提示 + 学生画像
**输出**：一句启发式追问（**红线：禁直接答案**）

```
你是苏格拉底式引导老师。学生需要提示，但你要用提问引导他自己思考，绝不直接给答案。

学生的问题：<question>{问题}</question>

已给过的提示（避免重复）：
<previous_hints>
{之前几轮给过的提示，无则写"无"}
</previous_hints>

学生画像：
<profile>
{掌握度分布}
</profile>

引导要求：
1. 只能用"提问"的方式引导，不能陈述答案
2. 每次只给一个提示性问题，不要一次问多个
3. 问题要指向"学生卡住的下一步"，而不是直接指向答案
4. 根据学生薄弱点调整提问方向
5. 【红线】绝对禁止：
   - 给出完整解法
   - 说出答案的关键步骤
   - 用陈述句陈述结论
   违反任何一条，重新生成

输出（自然语言，单句提问，不要JSON）：
{一个启发式追问}

示例（学生问"为什么AVL树查找快"，BST薄弱）：
✅ "你想想，如果一棵二叉搜索树退化成了链表，它的查找会变成什么复杂度？"
❌ "因为AVL树保持平衡所以查找是O(logn)"（这是陈述答案，违规）
```

**设计点**：
- **红线内置在 prompt 里**：禁完整解法/禁关键步骤/禁陈述句。这是 socratic_hint 这个 Skill 的核心校验
- 用 ✅/❌ few-shot 对比，让 LLM 直观理解"什么算给答案"
- 单句提问，防止一次问太多学生反而乱

---

## F. 在线-记忆与守门（2 个）

### Prompt 11：mastery_update_skill（认知状态更新）

**被谁调用**：诊断 Agent 完成诊断后（或每轮收尾系统处理）
**输入**：本轮诊断结果 + 当前画像
**输出**：JSON（各知识点掌握度的新数值）

```
你是学生认知状态更新器。根据本轮互动的诊断结果，更新学生在各知识点上的掌握度。

本轮诊断结果：
<diagnosis>
{诊断Agent或evidence_check_skill的结论，例：学生在二叉搜索树上答错}
</diagnosis>

学生当前画像：
<current_profile>
{各知识点的当前掌握度，0~1}
</current_profile>

更新规则：
1. 答对/理解 → 该知识点掌握度上调（如 +0.1~0.2）
2. 答错/卡住 → 该知识点掌握度下调（如 -0.1~0.2）
3. 只更新本轮有明确证据的知识点，没证据的不动
4. 掌握度范围 [0, 1]，不越界
5. 调整幅度根据证据强度：多次答错下调更多

输出 JSON：
{
  "updates": [
    {"knowledge_point": "知识点", "new_level": 0.0~1.0, "delta": "变化量(+/-)"}
  ]
}

示例：
{
  "updates": [
    {"knowledge_point": "二叉搜索树", "new_level": 0.15, "delta": "-0.05"},
    {"knowledge_point": "AVL树", "new_level": 0.25, "delta": "-0.05"}
  ]
}

约束：
- new_level 必须 0~1
- 只列有更新的知识点
- 只输出 JSON
```

**设计点**：
- **只更新有证据的节点**——防止画像被乱动画成噪声
- 量化规则（±0.1~0.2）是初始预设，可调

---

### Prompt 12：compliance_check（合规守门）

**被谁调用**：守门节点（强制，任何 Agent 输出返回调度后、发给用户前必过）
**输入**：拟输出的内容
**输出**：JSON（通过/拦截 + 原因）

```
你是教学合规审查员。下面是系统拟输出给学生的内容，请审查是否合规。

拟输出内容：
<output>
{任何Agent生成的内容}
</output>

审查红线（违反任何一条则拦截）：
1. 【泄答案】在考试/测验场景下，直接给出了题目答案或完整解法
2. 【敏感话术】含歧视、不当、伤害性表达
3. 【幻觉】内容里有超出证据范围的编造说法（标注了"建议查阅"的除外）
4. 【越界】给出了超出《数据结构与算法》课程范围的不当承诺（如保证提分等）

输出 JSON：
{
  "passed": true/false,
  "violated_rule": "违反的红线名（passed=true时为null）",
  "reason": "拦截原因（passed=true时为'通过'）",
  "suggestion": "passed=false时，建议如何修改"
}

示例（考试场景下泄了答案）：
{
  "passed": false,
  "violated_rule": "泄答案",
  "reason": "当前为考试场景，但输出包含完整解题步骤",
  "suggestion": "改为启发式提问，引导学生自己得出答案"
}

约束：
- passed=true 时其他字段可为null
- 只输出 JSON
```

**设计点**：
- **强制守门**——不归任何 Agent 管，系统级强制执行，这是合规底线
- 红线四类对应不同风险，suggestion 字段让拦截后有改进方向

---

## 全部 12 个 prompt 完成。优先级与测试建议

| 优先级 | Prompt | 建议立即测试 |
|---|---|---|
| **P0 核心** | 5 ReAct主控、7 evidence_check_skill | 复制进对话框，造一个"证据不全"的场景，看 gap_type 和 confidence 判断对不对 |
| **P1 重要** | 4 调度决策、8 answer_generate_skill、2 知识点抽取 | 用真实课件段落测抽取；造"新手vs熟练"两个画像测讲解难度适配 |
| **P2 辅助** | 6 query_rewrite_skill、9 weakness_diagnosis_skill、10 socratic_hint_skill | 造"对比题"测rewrite；造"卡在AVL"测diagnose能否追根因到BST |
| **P3 守门** | 11 mastery_update_skill、12 compliance_check_skill、1 图描述、3 题目转录 | 造"泄答案"场景测compliance能否拦截 |

**面试价值**：这 12 个 prompt 是你"真做过"的最强证据。面试官问"你的 prompt 怎么设计的"，你能直接说"我有完整的 prompt 设计，比如 evidence_check_skill 这个 prompt 我把每个 chunk 的 rerank_score 都写进去让 LLM 参考防止过度自信"——这种细节是装不出来的。

---

## 附：Qwen3-Reranker 调用格式（非 prompt）

Qwen3-Reranker 是判别模型，不是生成式 prompt。它的调用是把 (query, document) 拼接后输入，直接输出相关度分数。调用格式（以 Qwen3-Reranker 为例）：

```
输入拼接格式：
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and provide a evaluation.<|im_end|>
<|im_start|>user
<Document> {chunk文本} </Document>
<Query> {学生问题} </Query><|im_end|>
<|im_start|>assistant
<think></think>

输出：yes/no token 的概率 → 转换为 0~1 的相关度分数
```

**面试提醒**：若被问"你的 reranker 怎么调用的"，答"判别模型不是 prompt，按官方拼接格式输入(query,doc)输出分数，分数随chunk传入后续 evidence_check_skill 做客观兜底"。

---
