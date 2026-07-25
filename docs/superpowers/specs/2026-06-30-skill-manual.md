# Skill 说明书（最终版）

> 版本：基于双 Agent 固定串行模型（教学 Agent + 评估 Agent）。
> Skill 判据：**封装性**——封装了底层 Tool（+ 逻辑），有清晰 I/O。和 prompt 无关。

---

## 一、全部 Skill：2 个

| Skill | 作用 | 封装了什么 | 谁调用 |
|---|---|---|---|
| `hybrid_retrieval` | 混合检索：BM25+向量并行→RRF融合→rerank重排→返回带分数的chunk列表 | `bm25_search` + `vector_search` + `rerank` 三个 Tool | **教学 Agent**（ReAct 检索时调） |
| `weakness_diagnosis` | 薄弱点诊断：拿教学 Agent 的 signals + 学生画像 → 沿更深依赖链追根因 → 输出 {weaknesses, root_cause, learning_advice} | `graph_lookup` Tool + LLM 追根因逻辑（Prompt 9） | **评估 Agent**（串行调用） |

---

## 二、Tool 清单

**Agent 直接调用的 Tool（2 个）**：

| Tool | 作用 | 访问什么 |
|---|---|---|
| `graph_lookup` | 查知识图谱 1 跳前置依赖 | Neo4j |
| `fetch_by_kp` | 按知识点精确取 chunk | Milvus |

**hybrid_retrieval 内部 Tool（3 个，不暴露给 Agent）**：

| Tool | 作用 |
|---|---|
| `vector_search` | 语义向量检索（Milvus） |
| `bm25_search` | 关键词检索（BM25） |
| `rerank` | 相关度精排（Qwen3-Reranker） |

`rerank` 不是 Agent 调用的独立 Tool——它是 hybrid_retrieval 内部固定流程的一步，Agent 不知道它存在。

---

## 三、教学 Agent 的工具箱（ReAct 可选动作，3 个）

| 动作 | 类型 | 做什么 |
|---|---|---|
| `hybrid_retrieval(query)` | Skill | 混合检索，返回带分数的 chunk |
| `graph_lookup(concept)` | Tool | 查 Neo4j 拿 1 跳前置依赖 |
| `fetch_by_kp(kp)` | Tool | 按知识点精确取 chunk |

**Agent 自己在 Thought/Observation/Final Answer 里完成的（不调 Tool/Skill）**：
- 查询改写："搜得不好，换个词" → 下一轮调 hybrid_retrieval 时传新 query
- 证据评估："现有 chunk 够不够？缺什么？要不要追深？" → 体现在下一轮决策
- 答案生成：证据够了，Final Answer 输出讲解
- 多跳决策：调 graph_lookup 拿到依赖 → Thought 判断"要不要继续追"

---

## 四、一张图：谁用谁

```
教学 Agent（ReAct）
  ├─ 调 hybrid_retrieval Skill ──→ [bm25 + vector + rerank] Tool
  ├─ 调 graph_lookup Tool ──→ Neo4j
  └─ 调 fetch_by_kp Tool ──→ Milvus

评估 Agent（串行）
  └─ 调 weakness_diagnosis Skill ──→ [graph_lookup] Tool + LLM 追根因

Tool 层（5 个）
  vector_search / bm25_search / graph_lookup / fetch_by_kp / rerank
```

**没有更多了。** 之前 12 个 Skill 的版本已被推翻——那是为了凑"Skill 库"概念而过度设计。
