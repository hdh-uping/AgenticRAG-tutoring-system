---
name: hybrid_retrieval
description: 混合检索教材内容，通过 BM25、向量检索、RRF 和 Reranker 返回带分数与页码的 chunk。概念、原理、公式、理解和比较问题应优先使用；图谱缺少代码或上下文时，也可用它补充教材实现证据。
---

# hybrid_retrieval

## 作用
从教材向量库中搜索与查询语义相关的段落。内部封装了四步检索流水线。

## 触发条件
- 学生问「是什么」「为什么」「怎么理解」「A 和 B 有什么区别」
- 需要从教材中查找概念解释、算法原理

## 不适用
- 需要精确的结构化信息（代码、复杂度、操作清单）时不应作为首选，应优先用 graph_lookup；图谱证据不完整时可用本 Skill 补充教材上下文
- 需要对比两个概念 → 建议 combine 本 skill + graph_lookup

## 内部流程
```
query → BM25(关键词) + Vector(语义) → RRF 融合(k=60) → Qwen3-Reranker-0.6B → top-5 chunks
```

### 各步骤说明
1. **BM25 关键词检索**：jieba 分词 + 数据结构术语词典 + 停用词过滤，召回精确匹配的 chunk
2. **向量语义检索**：Qwen3-Embedding-0.6B 将 query 编码为 1024 维向量 → Milvus 余弦相似度搜索
3. **RRF 融合**：双路召回按排名融合（不依赖分数量纲），k=60
4. **Reranker 精排**：Qwen3-Reranker-0.6B Cross-Encoder 对每个 (query, doc) 联合编码打分，sigmoid 归一化到 [0,1]

### 输出格式
每个 chunk 带：
- rerank_score (0~1，越高越相关)
- 页码（教材对应位置）
- 章节路径
- 原文前 400 字

## 依赖
- Milvus Lite：`kb/vectordb/milvus.db`，266 chunks
- Qwen3-Embedding-0.6B：通过 `EMBED_MODEL_PATH` 配置
- Qwen3-Reranker-0.6B：通过 `RERANK_MODEL_PATH` 配置
- Chunk 数据：`kb/data/chunks_*.jsonl`

## Agent 调用方式
```
Action: hybrid_retrieval[查询内容]
```
例如：`hybrid_retrieval[单链表和顺序表插入操作的区别]`
