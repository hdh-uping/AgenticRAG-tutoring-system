---
name: graph_lookup
description: 查询知识图谱中的结构化知识，返回概念描述、伪代码、完整 C 代码、复杂度、关联操作和 IS_A 层级。具体操作、代码、复杂度、分类或关系问题应优先使用；若图谱信息不完整，再用 hybrid_retrieval 补充教材证据。
---

# graph_lookup

## 作用
优先从 Neo4j 精确查询概念；Neo4j 未配置或不可用时自动读取仓库内 JSON 图谱。图谱节点是完整知识单元，包含描述、算法步骤、C 代码和复杂度，不需要回查 chunk。

## 触发条件
- 学生问「有哪些操作」「代码怎么写」「时间复杂度」「属于哪类结构」
- 需要精确的结构化事实，不是模糊的语义搜索

## 不适用
- 概念解释、原理分析、「为什么」→ 用 hybrid_retrieval
- 需要对比两个概念 → 分别调本 skill 两次，或结合 hybrid_retrieval

## 内部流程
```
概念名 → 模糊匹配 Neo4j 节点（处理 · 分隔符变体）→ 查关联信息 → 格式化输出
```

### 各步骤说明
1. **构造搜索变体**：处理 `·` 分隔符问题。Agent 可能传 "顺序表插入"（无 ·），但图里存的是 "顺序表·插入"。自动生成变体重试。
2. **模糊匹配**：`WHERE n.name CONTAINS $kw` 匹配节点
3. **查关联信息**：
   - 如果匹配到**数据结构**节点 → 查 HAS_OPERATION（操作列表）、IS_A（父子类型）
   - 如果匹配到**操作**节点 → 输出 description + pseudocode + code + HAS_COMPLEXITY（时间复杂度）
4. **格式化输出**：结构化文本，标注节点类型和完整字段

### 输出格式
```
[操作] 顺序表·插入
  ℹ️  以上已包含完整代码和步骤，可直接用于回答。
  描述: ...
  步骤: 1. ... 2. ...
  代码: int ListInsert_Sq(...) { ... }
  时间复杂度: O(n)
```

如果匹配到数据结构节点，还会额外列出：
- 操作列表
- 子类型 / 父类型（IS_A 层级）

## 图谱数据
- 24 个数据结构节点
- 69 个操作节点：操作可带 description/pseudocode/code
- 3 个复杂度节点
- 116 条关系：IS_A (18) + HAS_OPERATION (69) + HAS_COMPLEXITY (29)

## 依赖
- Neo4j（可选）：通过 `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD` 配置
- 本地降级：`kb/graph/entities.json` + `kb/graph/relations.json`
- 图谱数据来源于 LLM 对 MinerU 解析结果的自动抽取

## Agent 调用方式
```
Action: graph_lookup[概念名]
```
例如：`graph_lookup[栈]`、`graph_lookup[顺序表·插入]`、`graph_lookup[队列]`
