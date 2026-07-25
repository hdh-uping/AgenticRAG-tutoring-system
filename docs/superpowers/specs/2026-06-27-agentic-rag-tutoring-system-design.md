# AgenticRAG 驱动的多 Agent 智慧教学辅导系统 — 设计文档（Spec）

> **文档性质**：这是一份"逻辑自洽、技术可行、将来可复现"的设计文档，不是"已跑通系统的总结"。
> 目标读者：项目作者（用于复现 + 面试准备）。
> 核心定位：**以 AgenticRAG 为深度锚点**的单课程教学知识引擎，其余模块为描述层，不装深度。
>
> **诚实声明（必须时刻记住）**：本文档描述的是设计意图与可复现依据，不是已验证的运行结果。简历上只能写"真做过的程度"。评测集的数字是唯一能低成本拿到的真实凭证——强烈建议实跑。

---

## 0. 关键设计决策速查表（所有锁定项）

| 决策项 | 选择 | 理由 |
|---|---|---|
| 投入程度 | 纯设计（不亲手跑代码） | 受限于时间；靠读论文/源码讲深度 |
| 技术锚点 | AgenticRAG（多跳检索 + 迭代推理） | Agent 岗核心考点，靠读能掌握 |
| 路由范式 | **C：全 Agent 决策（ReAct）** | 兑现"Agentic"卖点，但需读透 prompt/终止条件 |
| 多 Agent 架构 | **双 Agent 固定串行**（教学+评估），无调度/引导/守门 | 流程固定代码串行调；引导是评估自然延伸（2.1） |
| 示例课程 | 《数据结构与算法》 | 知识结构清晰，多跳问题好造 |
| 多模态 | 进主体（VLM 入口适配器） | 真实教学场景，亮点 +1 |
| 迭代判定 | LLM 自判证据充分性 | 简单好讲；不足则看"缺口类型"决定下一步 |
| 多跳触发 | 学生画像驱动 | 掌握度低才补前置知识 |
| 记忆节奏 | 短期实时 / 长期会话末异步写回 | 开销小、好讲 |
| 存储 | Milvus + Neo4j + SQLite（**无 Redis**） | 三件套，零冗余 |
| 知识图谱+画像 | Neo4j | 图查询是知识图谱岗加分标签 |
| 嵌入 + 重排 | Qwen3-Embedding + Qwen3-Reranker | 全 Qwen 生态自洽 |
| 作业批改 Skill | 砍掉 | 纯设计撑不住 OCR 批改流水线 |
| 评测数字 | 给预估范围，实跑时填入 | 方向必然成立，数字待实测 |

---

## 1. 项目定位与范围边界

### 1.1 一句话定位

面向单门课程（《数据结构与算法》），构建一个以 AgenticRAG 为核心的教学知识引擎：Agent 能自主规划检索路径，对课程资料做多跳、多轮的知识推理与校验，为答疑、学习诊断提供可靠的知识支撑。

### 1.2 做什么（IN）

- **单课程知识库构建**：课件 PDF/Markdown → 结构化解析 → 知识点抽取与关系图 → 混合检索索引
- **AgenticRAG 主链路**：查询改写 → 混合检索（BM25 + 向量）→ RRF 融合 → 重排 → LLM 自判证据充分性 → 不足则按"缺口类型"迭代检索
- **多跳场景**：学生画像掌握度低时，沿知识图谱补全前置知识
- **多模态入口**：学生上传题目图片 → VLM 转录成文字 + LaTeX → 进入标准 AgenticRAG 流程
- **双 Agent 固定串行**：教学 Agent（ReAct 检索+生成）+ 评估 Agent（诊断+建议）→ 系统拼回复（2.1）
- **小评测集**：20~50 条人工 QA（单跳/多跳/迭代三类）+ 检索策略对比表

### 1.3 不做什么（OUT，诚实边界）

- ❌ 不做"全院多课程规模化落地"（实习个人项目撑不住）
- ❌ 不做 Agent 间自治协商/对话（只做固定串行：教学→评估）
- ❌ 不做三层记忆（短期/工作/长期）——砍掉"工作记忆"层，只做短期 + 长期
- ❌ 不做作业 OCR 批改流水线
- ❌ 不做基于图片内容的多模态图片库检索（VLM 只做"图→文字"转录）

### 1.4 范围的诚实价值

面试官问"怎么处理多课程"时，诚实回答："单课程先打通，多课程是知识库的横向扩展，工程上是同一套流水线换数据源，我没有在这一期做。"——有取舍的判断，胜过吹"规模化落地"。

---

## 2. 系统架构总览

### 2.1 架构：双 Agent 固定串行（教学 + 评估）

```
学生输入(文字/图片)
  │
  ▼
① 系统预处理（代码，非 Agent）
   - VLM 转录 / 范围检查(知识库外→拒答) / 加载学生画像
   │
   ▼
② 教学 Agent ──── ReAct 循环(≤6轮),LLM决策 ────────────┐
   │  工具箱(3个,按需调):                                 │
   │    hybrid_retrieval(query)  Skill ─→ [bm25+vector+rerank]
   │    graph_lookup(concept)    Tool  ─→ Neo4j查1跳前置
   │    fetch_by_kp(kp)          Tool  ─→ Milvus精确取chunk
   │                                                      │
   │  Agent思考中完成(不占轮次):                            │
   │    · 查询改写:"上次搜得不好,换个词"                    │
   │    · 证据评估:"现有资料够不够?缺什么?"                  │
   │    · 多跳决策:"要不要沿依赖链再追一跳?"                 │
   │    · 答案生成:"证据够了,输出讲解"                       │
   │                                                      │
   │  返回: {answer, signals: {gap_types, missing_concepts,│
   │          final_confidence, total_iterations}}         │
   └────────────────────┬──────────────────────────────────┘
                        ▼
③ 评估 Agent ──── 串行,无ReAct ──────────────────────────┐
   │  weakness_diagnosis(signals, profile)  Skill         │
   │    ─→ [graph_lookup] Tool 查更深层依赖链追根因       │
   │    ─→ LLM推理: "根因在哪?学习建议是什么?"            │
   │    ─→ mastery_update 更新画像(内存立即可见+Neo4j异步) │
   │  返回: {weaknesses, root_cause, learning_advice}     │
   └────────────────────┬──────────────────────────────────┘
                        ▼
④ 系统拼最终回复（代码）
   answer + diagnosis + learning_advice → 一段自然回复
   │
   ▼ 返回学生
```

**没有调度 Agent**：流程固定（②→③），按顺序调就行，代码实现，不需要 LLM 决策"派谁"。
**没有引导 Agent**："指导"是评估 Agent 输出的一部分，系统拼回复时自然融入。
**没有合规守门**：教学答疑场景没有泄答案/敏感话术的业务需求，幻觉由证据评估兜底。

**技术分层**：
- 接入层：FastAPI + VLM转录
- Agent 层：教学 Agent（ReAct）+ 评估 Agent（串行）
- 能力层：Skill（2个）+ Tool（5个）
  - **Skills**（封装了 Tool 的复合能力，2 个）：
    - `hybrid_retrieval`：封装 bm25_search + vector_search + rerank 三个 Tool
    - `weakness_diagnosis`：封装 graph_lookup Tool + LLM 追根因逻辑
  - **Tools**（Agent 直接调用的原子外部访问，**2 个 Agent 可见**）：
    - `graph_lookup` / `fetch_by_kp`
    - `rerank`、`vector_search`、`bm25_search` 是 hybrid_retrieval Skill **内部实现**，不直接暴露给 Agent
- 存储层：Milvus + Neo4j + SQLite + LangGraph State

### 2.2 存储分工（明确责任）

| 数据 | 存储 | 寿命 | 读/写时机 |
|---|---|---|---|
| **向量**（chunk embedding） | **Milvus** | 永久 | 建库时写；检索时读 |
| **知识图谱**（知识点+前置依赖） | **Neo4j** | 永久 | 建库时写；多跳检索时读 |
| **学生画像**（掌握度） | **Neo4j**（学生节点-[:掌握]->知识点节点） | 永久 | 新会话拉取；会话末异步更新 |
| **会话历史**（对话流水） | **SQLite**（messages 表） | 永久 | 每轮追加；会话恢复时读 |
| **推理状态**（本轮子问题、候选chunk、自判） | **LangGraph State（内存）** | 一次问题处理期间 | 流转中读写；处理完即弃 |

**没有 Redis**：对话上下文存 SQLite，推理状态住内存，学生画像存 Neo4j。个人项目不需要 Redis 的共享缓存和 TTL 过期能力，引入只会增加被追问的技术点。

### 2.3 核心概念澄清（面试防混）

> **「短期记忆 ≠ 聊天记录」**
> - 聊天记录 = 对话事件流水 = 持久数据（存 SQLite，关掉重开可见）
> - 短期记忆 = Agent 处理当前问题时的临时工作状态 = 易失（LangGraph State/内存）
>
> **「Skill ≠ Tool，判据是封装性」**
> - Tool = Agent 直接调用的原子外部访问（vector_search、graph_lookup、fetch_by_kp、rerank…）
> - Skill = 封装了底层 Tool（+ LLM 逻辑），有清晰 I/O 的复合能力（hybrid_retrieval 封了 3 个 Tool、weakness_diagnosis 封了 graph_lookup + 追根因逻辑）
> - **全部 2 个 Skill、2 个 Agent 可见 Tool**（graph_lookup / fetch_by_kp）。rerank/vector_search/bm25_search 是 hybrid_retrieval 内部实现。Agent 可以同时调 Skill 和 Tool，不做限制。
>
> **「双 Agent 固定串行，非 Supervisor 调度」**
> - 教学 Agent（ReAct 检索+生成）→ 评估 Agent（诊断+建议）→ 系统拼回复。流程固定，代码按顺序调用，不需要调度 Agent 做 LLM 决策。
> - 评估 Agent 读取教学 Agent 返回的 signals（gap_types / missing_concepts / confidence），沿更深层依赖链追根因。
> - 画像更新：内存立即可见（同会话后续轮次受益）+ Neo4j 异步持久化。

---

## 3. 存储设计

### 3.1 Milvus — 向量库

Collection: `course_chunks`

| 字段名 | 类型 | 说明 | 支撑的检索行为 |
|---|---|---|---|
| `chunk_id` | INT64 (主键) | 自增 | 标识 |
| `embedding` | FLOAT_VECTOR(1024) | Qwen3-Embedding | **向量召回**（HNSW 索引） |
| `text` | VARCHAR | chunk 原文 | 检索后喂给 LLM；BM25 索引源 |
| `source` | VARCHAR | 来源文件名 | metadata |
| `page` | INT64 | 页码 | metadata |
| `source_kp` | VARCHAR | **核心知识点** | **补全取料精确过滤**（倒排索引） |
| `knowledge_points` | VARCHAR[] | 相关知识点数组 | 召回后处理过滤；建图时关联 |
| `difficulty` | INT64 | 难度等级 | metadata |

**字段设计的核心原则**：字段是用户自定义的（Milvus 只强制要求一个向量字段），每个字段对应一种检索/过滤行为，**没有冗余**。

**`source_kp` vs `knowledge_points` 的使用边界**：
- `fetch_by_knowledge_point`（多跳补全取料）用 `source_kp == kp` **精确过滤** → 保证取的是核心讲解 chunk，准不要多
- "沾边的"在首轮向量检索时已被召回，补全这一步不重复

**`source_kp` / `knowledge_points` 的数据来源**：不是天上掉的，是**建库阶段 LLM 知识点抽取**算出来的。存储结构和建库流程耦合，一荣俱荣一损俱损。

### 3.2 Neo4j — 知识图谱 + 学生画像

同一份课程资料的两种抽象：Milvus 存"文字块"，Neo4j 存"概念关系"。

**节点**：
```
(:知识点 {name, difficulty})          // 知识点概念
(:学生 {student_id})                  // 学生
```

**边**：
```
(:知识点)-[:前置 {depth}]->(:知识点)   // PREREQUISITE 依赖（多跳的核心）
(:知识点)-[:属于]->(:知识点)           // 知识点归属
(:学生)-[:掌握 {level: 0~1, updated_at}]->(:知识点)   // 学生画像
```

**为什么图谱和画像放一个库**：多跳诊断依赖 join——一句 Cypher 同时遍历依赖边 + 读掌握度：

```cypher
MATCH (kp:知识点 {name:'AVL树'})-[:前置*1..2]->(pre)
MATCH (s:学生 {id:'张三'})-[m:掌握]->(pre)
WHERE m.level < 0.5
RETURN pre.name, m.level
```

**Milvus 与 Neo4j 的协作**（核心设计点）：
- Milvus 回答"哪段文字相关"——靠语义相似，模糊匹配
- Neo4j 回答"还需要哪些概念"——靠逻辑依赖，精确推理
- 两个靠 chunk 的 `source_kp` / `knowledge_points` 标注连接

### 3.3 SQLite — 会话历史

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,        -- 'user' | 'assistant' | 'tool'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON              -- 可选：存检索片段、诊断结果等
);
CREATE INDEX idx_session ON messages(session_id);
```

会话历史是**持久数据**，关掉重开天然可见。

### 3.4 LangGraph State（内存）— 短期记忆

```python
class RAGState(TypedDict):
    question: str                    # 原始问题（多模态转录后）
    reasoned_queries: list[str]      # Agent Thought 中改写的查询
    candidates: list[Chunk]          # 当前累积的候选 chunk
    sufficiency: dict                # Agent 的证据评估结果
    gap_type: str                    # "缺前置知识" | "检索词不准" | None
    missing_concepts: list[str]      # 多跳推导出的前置知识点
    student_profile: dict            # 从 Neo4j 拉取的学生画像
    iteration: int                   # 当前迭代轮数（≤6）
    final_answer: str                # 最终答案
    signals: dict                    # 教学Agent返回给评估Agent的信号
```

**寿命 = 处理一个问题的耗时**（几秒~几十秒），处理完即弃。

---

## 4. 建库流程（离线，只跑一次）

### 4.1 流水线

```
课件 PDF / Word / PPT
        │
        ▼
┌────────────────────────────────────────────────────┐
│ 阶段1: MinerU 版面分析 + 分类处理                    │
│   正文/标题 → 文字（保留标题层级）                    │
│   公式 → LaTeX                                       │
│   表格 → Markdown 表格（行列结构保留）                │
│   图片 → 提取成图片文件 + 占位引用                    │
│   产出: 结构化 Markdown 文档                          │
└────────────────────┬───────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────┐
│ 阶段2: 图片内容描述（MinerU 干不了，必须额外做）       │
│   每张图片 → Qwen-VL → 文字描述                       │
│   描述回填到 Markdown 里图片占位的位置                 │
│   例: ![示意图](...) → "【图】AVL树LL型右旋示意图:..." │
└────────────────────┬───────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────┐
│ 阶段3: 语义切分（按知识点边界，非固定字数）            │
│   一个 chunk ≈ 一个可独立解释的知识单元                │
│   表格作为整体，不被切断                                │
└────────────────────┬───────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────┐
│ 阶段4: LLM 知识点抽取（给 Milvus 填 metadata）        │
│   每个 chunk → LLM 抽取:                              │
│     source_kp = "AVL树旋转"  （核心知识点）            │
│     knowledge_points = ["AVL树旋转","平衡因子"]        │
│   同时用这些知识点建 Neo4j 图谱节点 + PREREQUISITE 边   │
└────────────────────┬───────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────┐
│ 阶段5: 向量化 + 写库                                  │
│   chunk.text → Qwen3-Embedding → embedding           │
│   写入 Milvus: {chunk_id, embedding, text,           │
│                 source_kp, knowledge_points, ...}     │
│   写入 Neo4j: 知识点节点 + 依赖边                      │
└────────────────────────────────────────────────────┘
```

### 4.2 MinerU 图表处理机制

MinerU 工作分三步：
1. **版面分析**（视觉检测模型）：识别页面上的区域类型（标题/正文/公式/表格/图片）
2. **分类处理**：每种区域走不同管线（公式→LaTeX化、表格→结构识别→Markdown、图片→提取文件）
3. **统一输出**：结构化 Markdown（按阅读顺序）

### 4.3 诚实警告：图片，MinerU 只能"提取"不能"理解"

MinerU 对图片只做到"切出来、存成文件"，看懂内容需要额外的 **VLM 描述**步骤（阶段2）。否则图片是知识库的盲区——向量检索永远搜不到。

### 4.4 表格处理决策

**方案甲（采用）**：表格整体进 chunk，不切断。
- 理由：教学场景学生常需精确查表，完整性优先于检索友好度。
- 备选方案乙（表格转自然语言摘要）不采用，因为丢失精确数据。

### 4.5 两个 VLM 用途必须分清（防混）

| 用途 | 阶段 | 目的 |
|---|---|---|
| 建库时描述课件图 | 离线、一次 | 让知识库能检索到课件里的图 |
| 在线转录学生题目图 | 在线、每次提问 | 把学生拍的问题图变成文字 |

两件事都用 VLM 但目的不同，简历/面试别混成一个"多模态能力"。

---

## 5. 在线检索与 Agent 决策（项目核心）

### 5.1 路由范式：全 Agent 决策（ReAct）

**这是项目最重要的设计选择**：检索路径不是写死的代码顺序，而是 Agent 运行时动态决定。这是"Agentic"卖点的兑现，也是面试最容易被深挖的地方。

对比：
- **范式 A（固定流水线）**：路由由程序员写死，每次走一样的路 → 朴素 RAG
- **范式 C（全 Agent 决策，采用）**：路由由 LLM 运行时决定，不同问题走不同路 → Agentic RAG

### 5.2 Agent 工具箱：Skill 与 Tool 的最终定义

**判据**：Skill = 封装了底层 Tool（+ 逻辑），有清晰 I/O；Tool = Agent 直接调用的原子外部访问。和 prompt 无关。

#### 教学 Agent 的工具箱（ReAct 可选动作，3 个）

| 名称 | 类型 | 做什么 | 内部封了什么 |
|---|---|---|---|
| `hybrid_retrieval(query)` | **Skill** | 两阶段"宽召回→精排序"：第一阶段 BM25+向量双路并行召回，RRF 融合粗排；第二阶段 Qwen3-Reranker 语义级精排，返回带分数的 top-k chunk | `bm25_search` + `vector_search` + `rerank` 三个 Tool |
| `graph_lookup(concept)` | **Tool** | 查 Neo4j，返回该概念的 1 跳前置依赖列表 | 无（本身就是原子 Tool） |
| `fetch_by_kp(kp)` | **Tool** | 按知识点名在 Milvus 精确过滤取 chunk | 无（本身就是原子 Tool） |

#### Agent 自己思考完成（ReAct Thought/Observation，不占轮次）

| 能力 | 教学 Agent 怎么完成的 |
|---|---|
| 查询改写 | 思考里判断"上次搜得不好，换个词搜"，下一轮调 hybrid_retrieval 时传新 query |
| 证据评估 | 拿到检索结果后，观察 chunk 的数量和质量，思考"现有资料够不够回答？缺什么？" |
| 多跳决策 | 发现缺前置知识后，调 graph_lookup 看依赖，再思考"要不要沿依赖链继续追？追多深？" |
| 答案生成 | 证据够了，Final Answer 输出讲解（难度适配画像） |

**关键**：查询改写、证据评估、多跳决策、答案生成 **不是独立 Skill**——它们发生在 Agent 的思考/观察/最终输出里。Agent 每轮调用的只有 3 个：hybrid_retrieval（拿 chunk）、graph_lookup（看依赖）、fetch_by_kp（取指定 chunk）。

#### 评估 Agent 调用的 Skill（1 个）

| 名称 | 类型 | 做什么 | 内部封了什么 |
|---|---|---|---|
| `weakness_diagnosis(signals, profile)` | **Skill** | 拿教学 Agent 返回的 signals（gap_types/missing_concepts）+ 学生画像 → 沿更深依赖链追根因 → 输出 {weaknesses, root_cause, learning_advice} | `graph_lookup` Tool + LLM 追根因 + 学习建议生成 |

#### 最终清单

| 类别 | 数量 | 清单 |
|---|---|---|
| **Skill** | **2 个** | hybrid_retrieval / weakness_diagnosis |
| **Tool（Agent 可见）** | **2 个** | graph_lookup / fetch_by_kp（rerank、vector_search、bm25_search 为 hybrid_retrieval 内部实现） |
| **Agent** | **2 个** | 教学 Agent（ReAct） / 评估 Agent（串行）

#### signals 提取（教学 Agent 传给评估 Agent 的输入）

signals 不从 Agent Thought 的模糊总结中提取。在 ReAct 循环结束后，由代码扫描本轮工具调用的**确定性记录**来组装，不依赖 LLM：

| signal | 来源 | 提取逻辑 |
|---|---|---|
| `gap_types` | 本轮调过的 Tool 列表 | 调了 `graph_lookup` → 追加"缺前置知识"；调了 `hybrid_retrieval(新query)` → 追加"检索词不准"；都没调直接答 → 空 |
| `missing_concepts` | `graph_lookup` 和 `fetch_by_kp` 的**参数** | 直接记录每次调用时传入的概念名，去重 |
| `final_confidence` | 最后一轮证据池里 top-k chunk 的 `rerank_score` | 取 top-k 均值的映射值，区间规范化到 0~1 |

这三条 signals 随教学 Agent 的 answer 一起传给评估 Agent，评估 Agent 据此决定沿哪些概念追更深层依赖链。

### 5.3 ReAct 决策循环

**「轮」（iteration）的定义**：
- **一轮 = 一次 Tool/Skill 调用**（hybrid_retrieval / graph_lookup / fetch_by_kp）
- **不计入轮数**：Agent 的 Thought（思考）/ Observation（观察检索结果）/ Final Answer（生成答案）。查询改写、证据评估、多跳深度决策发生在 Thought 里，不占轮次。

```
循环（≤6 轮）:
  拿到本轮 Observation（检索结果/chunk/分数）
  → Thought: 证据够吗？缺什么？要不要换个词搜/沿依赖追深？
  → Action: 调 hybrid_retrieval(新query) / graph_lookup(概念) / fetch_by_kp(kp)
  → 或 Action: FINAL_ANSWER → 输出讲解 + signals
终止: 证据够了 → FINAL_ANSWER；或 6 轮到了 → 降级输出

循环结束后的代码后处理:
  signals = {
    gap_types: 本轮工具调用中出现的缺口类型（确定性提取，见5.2 signals表）,
    missing_concepts: 本轮传给 graph_lookup/fetch_by_kp 的概念参数去重,
    final_confidence: 证据池 top-k chunk 的 rerank_score 均值规范化
  }
  → 连同 answer 一起传给评估 Agent
```

**为什么是 6 而不是更大**：社区 ReAct/iterative retrieval 典型上限 3-5 轮；本系统放宽到 6 以应对教学场景的多层前置依赖（最复杂的多跳链路约 3-4 层，留 2 轮余量给 rewrite/重检索）。**不设更大**（如 20）是为了控制成本（每轮 ≈ 3 次 LLM 调用，6 轮 ≈ 18 次）和防止 Agent 反复打转。面试答"为什么是 6"：主流 3-5，放宽到 6 覆盖深多跳，再大就烧钱且偏离主流。

### 5.3.1 兜底降级 + confidence 真实来源（★ Agent 工程核心）

**问题：6 轮跑完，LLM 仍判断证据不充分，怎么办？**

**核心原则**：绝不无限加轮、绝不硬编答案。必须走**有控制的降级**。这是 Agent 工程的核心能力，也是面试区分"懂工程"和"只会吹 ReAct"的追问点。

**两条约束**：
- **成本可控**：6 轮还不行说明问题超出系统能力边界，应降级而非继续烧钱
- **诚实 > 逞强**：证据不足硬编答案（幻觉）在教学场景是灾难，宁可诚实标注/拒答

**Agent 在 Thought 中的证据评估输出**（降级分流用）：

```python
{
  "sufficient": bool,
  "gap_type": "缺前置知识" | "检索词不准" | "知识库外" | None,
  "confidence": float,        # Agent 对现有证据的把握度
  "reasoning": str
}
```

**confidence 的真实来源**：Agent 在 Thought 里自评把握度，结合每个 chunk 的 rerank_score（高分的多→更自信，高分的少→下调）。不是公式，是 LLM 主观自评 + 客观分数校正。

- confidence 不是数学公式 `f(x)`，是"**LLM 主观自报 + 客观重排分数校正**"的混合信号
- **阈值 0.4/0.7 是初始预设**，需在评测集上靠实测调优

**6 轮跑完后的降级分流表**：

| 最终评估 | 系统行为 |
|---|---|
| 不充分，confidence 中(0.4~0.7) | **尽力答 + 诚实标注** |
| 不充分，confidence 低(<0.4)，gap=知识库外 | **坦诚拒答** |
| 不充分，标记为"高价值难题" | **转人工**（可选） |

**面试深水区预警**：面试官追"阈值怎么定的"。诚实答：阈值初始预设，待评测集实测调优。**主动承认 LLM 自评有校准误差（calibration）**。

### 5.4 决策依据表（面试核心答案）

Agent 在 ReAct 思考环节决定下一步。注意：**Agent 工具只有 3 个（hybrid_retrieval、graph_lookup、fetch_by_kp）。查询改写、证据评估、多跳深度——都是 Thought 里完成的。**

| 当前状态 | Agent 思考 | 行动 |
|---|---|---|
| 刚拿到问题 | "先查资料" | 调 `hybrid_retrieval(query)` |
| 拿到检索结果，chunk 少或 rerank_score 低 | "证据不够，缺什么？是不是检索词不好？" → 换个词 | 调 `hybrid_retrieval(新query)` |
| 证据不够，但判断是缺前置知识 | "这题需要前置概念，看看依赖链" | 调 `graph_lookup(concept)` → 拿到前置列表 |
| 拿到前置列表，发现某前置画像掌握度低 | "这个前置知识需要补" | 调 `fetch_by_kp(前置kp)` → 取它的 chunk |
| 拿到前置 chunk 后 | "要不要沿依赖链再追深一跳？" → 追 | 调 `graph_lookup(那个前置概念)` |
| 拿到前置 chunk 后 | "够了，追下去没必要" → 停 | **直接进入 Final Answer** |
| 证据充分 | "够回答了" | **Final Answer 输出讲解** |
| 6 轮还没充分 | "先降级输出" | **Final Answer 输出讲解 + 诚实标注不确定性** |

**关键**：evicence 评估不是独立 Skill，发生在 Agent 的 Observation 后——"刚检索回来的结果，看看质量如何"→ 体现在下一轮调 hybrid_retrieval 的新 query 里。

### 5.5 检索链路（ReAct 内部，一条典型轨迹）

```
学生问:"B+树为什么比B树更适合做数据库索引?"

轮1: Thought"先检索" → 调 hybrid_retrieval(问题)
     Obs: top5 chunk, 各有score, 感觉缺对比B树的资料
     Thought"缺B树信息,B+树依赖B树" → 调 graph_lookup("B+树")
     Obs: 前置 [B树, 磁盘IO]
     画像判断: B树掌握度低 → 需要补

轮2: Thought"取B树的chunk" → 调 fetch_by_kp("B树")
     Obs: 拿到B树结构讲解
     Thought"还要不要继续追B树的前置?" → LLM判断: B树对比索引已经够清楚了,不追

轮3: Thought"证据够了" → Final Answer 输出讲解
     附带 signals: {gap_types:["缺前置知识"], missing_concepts:["B树"], confidence:0.85}
     ↑ 这些 signals 是检索过程的自然副产品,传给评估Agent
```

**注意**：全程只调了 3 次 Tool（hybrid_retrieval × 1 + graph_lookup × 1 + fetch_by_kp × 1 = 3 轮检索轮次）。查询改写（如果第一轮搜得不好换个词）发生在 Thought 里，下一轮调 hybrid_retrieval 时传新 query 就行，不需要独立 Skill。
   ▼
合并 chunk → 重排 → 回到自判（≤6 轮）
```

### 5.6 关键澄清：BM25 / 向量 / 补全检索的分工

- **BM25（稀疏）**：只在首轮混合检索出现，靠关键词精确匹配
- **向量（Milvus，密集）**：首轮混合检索的另一半，靠语义相似
- **补全检索（fetch_by_kp）**：是 **`source_kp` 精确过滤**，不重新算向量相似度，也不做 BM25

**两维别混**：
- 手段维（向量 vs BM25）→ 回答"用什么算法找"
- 时机维（首轮 vs 补全）→ 回答"第几轮去找"；补全换的是 query 不是手段

### 5.7 选 C 的诚实预警（必须读透）

- ReAct 的 prompt 设计必须读透（LangChain/LangGraph 标准内容）
- 最大轮数为什么是 6、如何防死循环、工具调用失败如何兜底，都要能答
- 可复现性下降：同一问题每次路径可能不同——主动说"这是我设计时纠结过的取舍"，是加分

---

## 6. 记忆系统

### 6.1 三层数据（按寿命）

| 层 | 数据 | 存储 | 寿命 | 读写 |
|---|---|---|---|---|
| 对话上下文 | 最近 N 轮对话，传给 LLM 的 messages 列表 | SQLite（messages 表） | 永久（全量）；调 API 时取最近 N 条 | 每轮追加；会话恢复读 |
| 推理工作区 | 本轮子问题、候选chunk、评估结果 | LangGraph State（内存） | 一次问题期间 | 流转中；答完即弃 |
| 学生画像 | 各知识点掌握度 | Neo4j | 跨会话永久 | 新会话拉取；评估后更新（内存立即可见 + Neo4j 异步持久化） |

### 6.2 为什么没有 Redis

对话上下文用 SQLite 足以覆盖个人项目的需求：查最近 N 条拼进 messages 列表传给 LLM。Redis 在生产环境多实例场景下有意义（共享缓存 + TTL 自动过期），但个人项目引入它只会增加运维负担和一个被追问的技术点。

---

## 7. Skill 库

> 完整定义见 5.2。此处展开 2 个 Skill 的细节。

### 7.1 Skill 的唯一定义

- **判据：封装性**。封装了底层 Tool（+ 逻辑）、有清晰 I/O → Skill；Agent 直接调的原子外部访问 → Tool。
- 全部只有 **2 个 Skill、2 个 Agent 可见 Tool**（graph_lookup / fetch_by_kp）。rerank 等是 hybrid_retrieval 内部步骤。
- Agent 可以调 Skill 也可以调 Tool，不做限制。

### 7.2 Skill 清单（2 个）

| Skill | 作用 | 封装了什么 | 谁调用 |
|---|---|---|---|
| `hybrid_retrieval` | BM25+向量并行→RRF融合→rerank重排→返回带分数的chunk列表 | `bm25_search` + `vector_search` + `rerank` 三个 Tool | 教学 Agent（ReAct 检索时调） |
| `weakness_diagnosis` | 拿教学 Agent 返回的 signals + 学生画像 → 沿更深层依赖链追根因 → 输出 {weaknesses, root_cause, learning_advice} | `graph_lookup` Tool + LLM 追根因逻辑 | 评估 Agent（串行调用） |

> **Agent 自己思考完成、不走 Skill 的**：查询改写、证据评估、多跳决策深度、答案生成。这些都发生在 ReAct 的 Thought/Observation/Final Answer 里，不是独立 Skill。

### 7.3 Skill 内部机制

**hybrid_retrieval**：两阶段"宽召回→精排序"
```
第一阶段·宽召回:
  parallel: bm25_search(query) + vector_search(query) → 各取 top_k
  → RRF 融合（不依赖分数量纲，BM25和向量分数范围不同，RRF按排名融合更鲁棒）

第二阶段·精排序:
  rerank(粗排候选, query) → Qwen3-Reranker 对每个(query,doc)联合编码打分
  → 按分数降序取 top_k，每个 chunk 带各自 rerank_score
  → 过滤低分噪音，只保留高分 chunk 进入后续 Agent 评估
```

**weakness_diagnosis**：
```
1. graph_lookup(教学Agent返回的 missing_concepts) → 拿更深层依赖链
2. 对比学生画像，标记掌握度低的节点
3. LLM 推理（Prompt 9）：根因在哪？学习建议是什么？
4. 输出 {weaknesses, root_cause, learning_advice}
5. mastery_update 更新画像（内存立即可见 + Neo4j 异步写回）
```

---

## 8. 端到端数据流

**学生问："B+树为什么比B树更适合做数据库索引？"**

```
① 系统预处理（代码）
   - VLM转录 / 范围检查 / 加载学生画像
   │
   ▼
② 教学 Agent ──── ReAct 循环（≤6轮）────
   轮1: Thought"查资料" → hybrid_retrieval(问题)
        Obs: 5个chunk各带score, top-1是"B+树结构"
        Thought"缺B树的对比资料,这可能涉及前置知识"
        调 graph_lookup("B+树") → Obs: 前置[B树, 磁盘IO]
        画像判断: B树 掌握度 0.2 → 需要补
   轮2: 调 fetch_by_kp("B树") → Obs: B树结构chunk
        Thought"现在可以对比B+树和B树了,证据够了"
        Final Answer: 输出讲解（难度适配画像）
	   ──── 返回: answer
	   │
	   ▼ 代码后处理提取 signals（扫描本轮工具调用记录,确定性提取,不靠LLM）
	   signals = {gap_types:["缺前置知识"], missing_concepts:["B树"], confidence:0.85}
	   │
	   ▼
	③ 评估 Agent
   weakness_diagnosis(signals, profile)
     → graph_lookup("B树") 追更深: B树→二叉搜索树→二叉树
     → 画像显示 BST 0.3 → LLM 推理:"根因是BST薄弱"
     → learning_advice:"建议回顾二叉搜索树的查找复杂度"
   mastery_update: BST 掌握度 0.3→0.25（异步写Neo4j）
   返回: {weaknesses:[{BST,0.7}], learning_advice:"..."}
   │
   ▼
④ 系统拼回复
   answer + "我注意到你对二叉搜索树的掌握可能不够..." + learning_advice
   │
   ▼ 返回学生
```

**关键**：
- 全程 2 次 Agent 调用，无调度 Agent、无引导 Agent、无合规守门。
- 教学 Agent 只调了 2 次 Tool（hybrid_retrieval + graph_lookup + fetch_by_kp = 3 轮，其中 graph_lookup 在轮1 后段调）。
- 证据评估发生在 Thought 里，不需要独立 Skill。多跳深度也是 Agent 自己决定的。
- 评估 Agent 读教学 Agent 返回的 signals 做诊断，教评分离。

---

## 9. 评测设计（唯一硬通货）

### 9.1 评测集结构

20~50 条人工 QA，覆盖《数据结构与算法》，分三类证明 Agentic 价值：
- **单跳直问**（~10 条）：如「快排时间复杂度」
- **多跳依赖**（~15 条）：如「为什么 B+ 树比 B 树更适合索引」←需要前置知识
- **迭代校验**（~10 条）：需要两轮检索才能凑齐证据的问题

### 9.2 对比实验表（数字实跑时填）

| 检索策略 | Recall@5 | 端到端准确率 | 多跳问题正确率 |
|---|---|---|---|
| 仅向量召回 | _（预估）_ | _ | _ |
| + BM25 混合 (RRF) | _ | _ | _ |
| + Qwen3-Reranker 重排 | _ | _ | _ |
| **+ 多跳补全（完整 AgenticRAG）** | _ | _ | _ |

### 9.3 预估范围（方向必然成立，数字待实测）

> ⚠️ 以下为基于 RAG 领域经验的**预期范围**，实跑时填入真实值。
> 方向依据：多跳策略在多跳问题上必然涨点；重排对 top-k 精炼必然有效；混合检索召回必然优于单路。

- 仅向量 → +混合检索：Recall@5 预期提升 5~15%
- +混合 → +重排：top-1 准确率预期提升 5~10%
- 完整 Agentic（+多跳） vs 朴素 RAG：**多跳类问题正确率预期提升 15~30%**（这是简历数字的主要来源）

### 9.4 评测手段

- **端到端准确率**：LLM-as-judge（GPT-4o 对比参考答案打 1-5 分）或人工评分
- **Recall@k**：人工标注的证据集

### 9.5 为什么是诚实的

方法论（多跳对多跳问题的提升、重排对 top-k 的精炼）是 RAG 领域公认有效，数字方向必然成立。哪怕只 20 条数据，也比没有数字的"我做了个系统"强十倍。

---

## 10. 技术栈（现代化版）

| 角色 | 选型 | 备注 |
|---|---|---|
| LLM | Qwen-2.5-14B（或更新，走 API / vLLM） | ⚠️ 核实当前版本号 |
| 嵌入 | **Qwen3-Embedding**（4B 或走 API） | ⚠️ 核实版本 |
| 重排 | **Qwen3-Reranker** | ⚠️ 核实版本 |
| VLM | Qwen-VL（API） | 建库描述图 + 在线转录题目 |
| 文档解析 | **MinerU**（替换 OCR） | 版面分析 + 结构化输出 |
| 编排 | LangGraph | 状态机 + ReAct |
| 向量库 | Milvus | chunk + 自定义字段 |
| 图谱+画像 | Neo4j | Cypher 多跳查询 |
| 会话历史 | SQLite | messages 表 |
| 后端 | FastAPI | 接口层 |
| 部署 | Docker | 容器化 |

**砍掉**：~~OCR~~、~~Redis~~（对话上下文用 SQLite 足够）、~~BGE-M3~~（改 Qwen3-Embedding）、~~工作记忆~~

**⚠️ 简历前必须自核**：所有 Qwen 系列的具体版本号、size 选择。本文档基于 2026 年中的认知，版本可能已更新。

---

## 11. 面试深挖防御地图

> 这一节专门为"纯设计、没跑代码"的你准备：每个面试官爱追的点，对应你能回答的层次。

| 面试官会问 | 你能答的层次 |
|---|---|
| "你的 Agentic 怎么实现的" | 教学 Agent 用 ReAct 循环做检索决策，工具只有 3 个（hybrid_retrieval、graph_lookup、fetch_by_kp）。查询改写/证据评估/多跳深度/答案生成都是 Thought 里完成的（5.4/5.5） |
| "Agent 怎么决定调哪个" | 每轮 Observation 后 Thought 判断：资料不够→调 hybrid_retrieval(换query)/graph_lookup(看依赖)/fetch_by_kp(取指定chunk)；够了→Final Answer（5.4） |
| "为什么不用固定流水线" | 固定流水线是朴素 RAG，我选 ReAct 是为了兑现"自主规划"——每跳追不追、追多深是 LLM 定的 |
| "你的 Agent 架构" | **双 Agent 固定串行**：教学 Agent（ReAct检索+生成）→ 评估 Agent（拿signals诊断+建议）→ 系统拼回复。没有调度Agent（流程固定，代码就够了）。教评分离（2.1） |
| "Skill 有几个" | **只有 2 个**：hybrid_retrieval（封3个Tool）、weakness_diagnosis（封graph_lookup+追根因）。其余都是 Agent 思考或 Tool。不为凑概念而加 Skill（5.2/7.2） |
| "Skill 和 Tool 怎么区分" | 判据是**封装性**：Skill 封装了底层 Tool(+逻辑)，Tool 是原子外部访问。Agent 可以同时调 Skill 和 Tool（5.2） |
| "为什么查询改写/证据评估不是 Skill" | 不需要封装 Tool，Agent 在 Thought 里就能想。把它们做成独立 Skill 是浪费轮次（5.4） |
| "6 轮跑完仍不充分怎么办" | 三条降级：confidence 中→尽力答+标注不确定性；confidence 低→坦诚拒答；高价值难题→转人工（5.3.1） |
| "confidence 怎么来的" | Agent 思考中自然评估，结合 rerank_score 客观校正。阈值 0.4/0.7 初始预设待实测调优（5.3.1） |
| "多跳深度怎么控制" | Agent 每跳后自己判断要不要继续追——画像掌握度高→停、证据够了→停、6轮上限→停。不是代码写死（5.4） |
| "为什么是6轮不是更少/更多" | 主流3-5轮，放宽到6覆盖深多跳+rewrite重试。教学场景每跳1-2轮（查依赖+取chunk），6轮够覆盖3跳深度（5.3） |
| "一轮是什么" | 一次调 Tool（hybrid_retrieval/graph_lookup/fetch_by_kp）。Thought/Observation/Final Answer 不算轮（5.3） |
| "知识图谱什么用" | 存 PREREQUISITE 依赖边。graph_lookup 查 1 跳，Agent 决定追不追深。评估 Agent 用它追更深（8） |
| "短期记忆是什么" | 推理中间状态住 LangGraph State；聊天记录持久住 SQLite（2.3/6.2） |
| "画像更新时机" | 内存立即可见（同会话后续轮次受益）+ Neo4j 异步持久化（6.2） |

**⚠️ 你答不出的层次（诚实避坑，被问到就主动承认待实测）**：
- confidence 阈值、reranker 参数具体怎么调的、bad case 怎么处理的（没跑代码）
- 部署性能、QPS、并发、延迟等工程指标
- 评测的具体数字（实跑前只有预估范围）

---

## 12. 复现路线（将来真做时）

按依赖顺序（自底向上）：

1. **建库**：MinerU 解析 → VLM 描述图 → 语义切分 → knowledge_extract(Prompt 2) 抽知识点 → Milvus + Neo4j
2. **Tool 层**：graph_lookup / fetch_by_kp（Agent 直接调）。（rerank/vector_search/bm25_search 为 hybrid_retrieval 内部实现）
3. **Skill 层**：hybrid_retrieval（封 3 个 Tool）+ weakness_diagnosis（封 graph_lookup + Prompt 9）
4. **Agent 层**：教学 Agent（ReAct，工具 3 个）+ 评估 Agent（串行调 weakness_diagnosis）
5. **记忆 + 多模态入口**：SQLite 会话 / Neo4j 画像 / VLM 转录（Prompt 3）
6. **评测**：造 QA 集 + 跑对比表 → 出数字

**纯设计路线最快上手**：把 Prompt 9 (weakness_diagnosis) 丢对话框测追根因逻辑，造一个"学生答错 AVL 但根因是 BST"的场景看输出。

---

## 附：与原介绍的差异（诚实记录）

| 原介绍 | 本设计 | 变更原因 |
|---|---|---|
| OCR 解析 | MinerU | OCR 丢版式/公式/表格，MinerU 是 2025+ 主流 |
| Cross-Encoder（笼统） | Qwen3-Reranker | 明确具体模型，全 Qwen 生态 |
| 未提嵌入模型 | Qwen3-Embedding | 补齐，全生态自洽 |
| 三层记忆（短/工/长） | 两层（短/长） | 纯设计撑不住"工作记忆"概念 |
| Redis | 砍掉 | 对话上下文 SQLite 足够；Redis 的生产特性（共享缓存/TTL）个人项目不需要 |
| 多 Agent 协同 | **双 Agent 固定串行**（教学+评估），砍掉调度/引导 Agent | 调度Agent是过度设计（流程固定，代码可串行调）；引导 Agent 鸡肋（"指导"是评估 Agent 输出的自然延伸） |
| Skill 体系 | 从原介绍 5 个→膨胀到 12 个→最终收缩到 **2 个真 Skill**（hybrid_retrieval、weakness_diagnosis） | 只有封了底层 Tool 的才叫 Skill。查询改写/证据评估/答案生成是 Agent 思考，不做成 Skill |
| 合规守门 | 砍掉 | 教学答疑场景无泄答案/敏感话术业务需求，幻觉由证据兜底 |
| 未提迭代上限 | 6 轮 ReAct 上限 + 降级分流 | 成本可控+防死循环+防幻觉（5.3/5.3.1） |
| "院内多课程规模化" | 单课程 | 实习个人项目边界 |
| 画像更新时机 | **内存立即生效 + Neo4j 异步持久化** | 旧版"会话末写回"导致同会话诊断白做；修正后同会话后续轮次立即可见（6.2） |
| BM25+向量（没说融合） | 明确 RRF 融合 | 是深度考点 |
| 作业自动化批改 | 砍掉 | 纯设计撑不住 OCR 批改流水线 |
