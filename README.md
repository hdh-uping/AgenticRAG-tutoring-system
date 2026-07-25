# Agentic RAG Tutoring System

面向《数据结构与算法》的教材溯源型教学 RAG。当前版本由教学 Agent、推荐 Agent 和三个 Skill 组成，重点是让检索、回答、推荐和会话形成一条可运行、可追踪的闭环。

## 当前架构

```text
POST /chat
  ├─ Bearer Token：从登录令牌确定用户身份和会话归属
  ├─ SQLite：持久化用户、登录令牌摘要、完整会话历史和跨会话偏好
  ├─ 双 Agent 父 LangGraph
  │   ├─ prepare_context
  │   ├─ 教学 Agent 子图：最多 3 轮结构化 ReAct
  │   │   ├─ prepare → decide ↔ execute_skill：计划、动态选 Skill 与取证
  │   │   ├─ hybrid_retrieval：BM25 + Qwen Embedding + RRF + Qwen Reranker
  │   │   ├─ graph_lookup：Neo4j 的 IS_A / HAS_OPERATION / HAS_COMPLEXITY
  │   │   └─ draft → verify → revise / verify_retrieval → finalize
  │   ├─ 推荐 Agent 子图：最多 2 轮独立决策
  │   │   └─ decide ↔ related_concepts → finalize / fallback
  │   └─ recommendation_fallback → assemble_response
  └─ SQLite：事务式保存问题和回答，更新会话标题与时间
```

教学 Agent 会先生成结构化任务计划，显式记录问题类型、必答要求、证据目标和完成检查，但不预先绑定具体 Skill。LangGraph 将 `prepare`、`decide`、`execute_skill`、`draft`、`verify`、`fallback` 和 `finalize` 等步骤组织成显式状态图；`decide` 仍由 LLM 根据各 Skill 的 `description` 和每轮 Observation 动态选择 `hybrid_retrieval` 或 `graph_lookup`。后端只校验动作是否属于已注册 Skill、是否取得有效证据以及答案是否覆盖任务要求。硬编码的 Skill 选择仅在模型未取得任何证据时作为异常兜底。Reranker 分数只判断相关度，不再直接触发结束。动作决策与最终 Markdown 正文分开生成，避免内部 JSON 泄漏；来源列表最多保留 3 条正文实际引用或最强证据。

三个 Skill 采用渐进式披露：发现阶段只读取并向对应 Agent 暴露 `name` 和 `description`；某个 Skill 首次被选择后，才读取它的 `SKILL.md` 正文并写入该 Agent 的后续上下文，同时动态导入对应的 `scripts/run.py`。未触发 Skill 的正文和执行模块不会预加载；同一进程内已经触发的模块会缓存复用。

草稿完成后，教学 Agent 内部执行一次验证阶段：代码先检查协议泄漏、函数实现、边界处理、复杂度和引用 ID，再由 LLM 对照任务计划与 Observation 审查证据覆盖。验证结果为 `pass`、`revise` 或 `retrieve_more`；后两者最多触发一次修订或一次补检索后重写，防止无限自我反思。验证是教学 Agent 的内部阶段，不新增第三个 Agent，因此系统仍保持双 Agent 架构。

父 LangGraph 负责“教学 Agent → 推荐 Agent → 回答组装”的整体编排。推荐 Agent 仍拥有自己的子图、提示词、状态和决策轨迹，先调用 `related_concepts` 获取图谱候选，再结合本轮问题与教学回答选择 0～2 个推荐，并为每项说明如何承接本轮学习内容；它不是确定性的推荐模块。多概念问题会均衡召回各概念的候选，避免只推荐比较中的一侧。它只能使用候选中的概念，不推断当前图谱不存在的 PREREQUISITE；推荐子图异常时，父图沿降级边保留教学回答，不让可选推荐拖垮主流程。

LangGraph State 只保存单次请求中的计划、消息、证据记录、动作轨迹和路由结果，处理完成后即释放；跨请求的完整会话和用户偏好仍由现有 SQLite 业务表负责。当前没有启用 LangGraph Checkpointer，因为项目已经有稳定的会话持久化接口，避免同时维护两套会话真相来源。三个 Skill 的数据来源仍限定为本地教材向量库和知识图谱。

SQLite 会永久保存每个会话的全部消息，以及教学 Agent、推荐 Agent 的轨迹。重新启动服务后，登录用户可以列出自己的会话、打开完整历史并继续提问。为避免无限增长的历史超过模型上下文，每轮只注入最近 `SESSION_CONTEXT_MAX_MESSAGES` 条且总计不超过 `SESSION_CONTEXT_MAX_CHARS` 字符；这个窗口不会删除数据库中的旧消息。

每次 `/chat` 会先用本地轻量规则判断本轮是否可能包含偏好；普通知识提问直接复用已保存偏好，只有出现长期倾向、自身水平或回答方式信号时才调用当前配置的 LLM 做语义提取。候选规则只负责路由，不决定具体偏好。提取结果经过固定字段和值域校验后才写入 SQLite；模型调用失败或返回非法格式时跳过本轮更新，不影响正常问答。偏好属于用户而不是单个会话，因此新会话第一轮也会注入教学 Agent 的 system prompt。带有“只限这次”等明确临时范围的要求不会持久化，也可以通过受 Bearer Token 保护的 `/prefs` 显式查看或修改。

## 数据规模

- 教材：线性表、栈和队列、数组
- 文本块：266 个，其中数组 27 个
- 图谱实体：96 个，包括 24 个数据结构、69 个操作、3 个复杂度
- 图谱关系：116 条，包括 18 条 IS_A、69 条 HAS_OPERATION、29 条 HAS_COMPLEXITY
- 向量库：Milvus Lite，本地文件位于 `kb/vectordb/milvus.db`

数组已经通过统一入库流水线处理：MinerU 生成 102 个标准 content 元素，VLM 转写正文插图，切分为 27 个带 1～5 页页码的 chunk。图谱来源统一为 `文档名:content:局部序号`，不再混用 `source_chunks`。

## 环境准备

推荐 Python 3.10～3.13。创建虚拟环境后安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

本机已经安装依赖的 Conda 环境为 `tec_stack`，使用它启动时先执行：

```bash
conda activate tec_stack
```

复制环境变量模板：

```bash
cp .env.example .env
```

运行完整问答至少需要填写：

- `LLM_API_KEY`
- `EMBED_MODEL_PATH`
- `RERANK_MODEL_PATH`

Neo4j 是可选增强。未配置 `NEO4J_PASSWORD` 或服务不可用时，系统会自动读取 `kb/graph/entities.json` 和 `relations.json`，图谱查询与关联推荐仍可工作。

已经出现在历史代码或聊天记录中的凭据应先在对应平台撤销并重新生成，不能继续使用。

## 启动

一键启动前后端（默认使用 `tec_stack` Conda 环境）：

```bash
./start.sh
```

浏览器访问 `http://127.0.0.1:5173`，按 `Ctrl+C` 会同时关闭前后端。首次运行且 `frontend/node_modules` 不存在时，脚本会自动执行 `npm install`。只检查环境但不启动服务可运行 `./start.sh --check`；端口等可配置项可用 `./start.sh --help` 查看。

也可以按下面的方式分别启动和调试前后端。

如果希望使用 Neo4j，先启动本机服务并确认已有图谱数据。若只需把已有 JSON 图谱重新导入：

```bash
python kb/scripts/build_graph.py --import-only --output-dir kb/graph
```

启动 API：

```bash
uvicorn app.main:app --reload
```

另开一个终端启动 React 前端：

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。开发环境中的前端请求统一发送到 `/api`，Vite 会将其代理到 `http://127.0.0.1:8000` 并移除 `/api` 前缀，因此本地联调不需要在 FastAPI 中额外开放 CORS。代理目标可在 `frontend/vite.config.ts` 中修改。

前端已实现注册与登录、会话新建/切换/重命名/删除、历史消息恢复、连续提问、教材来源查看、Agent 执行轨迹展开、用户偏好设置及移动端布局。登录令牌保存在当前浏览器的 `localStorage`，注销或后端返回 401 时会清除。

检查前端生产构建：

```bash
cd frontend
npm run build
```

检查服务：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`/ready` 会检查 Python 运行依赖、模型权重、本地图谱，并实际打开 Milvus collection 检查记录数；第一次检查可能需要等待 Milvus Lite 冷启动。它不会调用 DeepSeek 或产生模型费用。

首次使用先注册；已有用户使用 `/auth/login`：

```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","password":"replace-with-a-strong-password"}'
```

响应中的 `access_token` 是后续请求使用的 Bearer Token。SQLite 只保存它的 SHA-256 摘要，不保存令牌原文。以下示例中的 `<访问令牌>` 均替换为该值。

提问示例：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Authorization: Bearer <访问令牌>' \
  -H 'Content-Type: application/json' \
  -d '{"question":"顺序表插入的时间复杂度是多少？"}'
```

首次 `/chat` 不传 `session_id` 时自动创建会话，并在响应中返回ID。继续同一会话：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Authorization: Bearer <访问令牌>' \
  -H 'Content-Type: application/json' \
  -d '{"question":"请结合刚才的回答再简单解释一次","session_id":"响应中的session_id"}'
```

会话管理接口：

```bash
# 列出用户会话
curl -H 'Authorization: Bearer <访问令牌>' \
  'http://127.0.0.1:8000/sessions'

# 打开会话并读取历史；长会话用 limit/offset 分页
curl -H 'Authorization: Bearer <访问令牌>' \
  'http://127.0.0.1:8000/sessions/会话ID?limit=200&offset=0'

# 显式创建空会话
curl -X POST http://127.0.0.1:8000/sessions \
  -H 'Authorization: Bearer <访问令牌>' \
  -H 'Content-Type: application/json' \
  -d '{"title":"线性表复习"}'

# 重命名会话
curl -X PATCH http://127.0.0.1:8000/sessions/会话ID \
  -H 'Authorization: Bearer <访问令牌>' \
  -H 'Content-Type: application/json' \
  -d '{"title":"新的标题"}'

# 删除会话及其全部消息（不可恢复）
curl -X DELETE -H 'Authorization: Bearer <访问令牌>' \
  'http://127.0.0.1:8000/sessions/会话ID'
```

偏好字段包括 `depth`、`show_code`、`style` 和 `response_length`：

```bash
curl -X PUT http://127.0.0.1:8000/prefs \
  -H 'Authorization: Bearer <访问令牌>' \
  -H 'Content-Type: application/json' \
  -d '{"depth":"beginner","show_code":"idea","style":"academic","response_length":"concise"}'
```

注销当前令牌：

```bash
curl -X POST http://127.0.0.1:8000/auth/logout \
  -H 'Authorization: Bearer <访问令牌>'
```

`/chat` 响应中的 `sources` 字段会列出本次实际使用的教材 chunk 或图谱节点。查看完整教材片段：

```bash
curl 'http://127.0.0.1:8000/sources/线性表_chunk_003'
```

`/health` 只表示 HTTP 进程存活；`/ready` 表示关键运行依赖、本地模型、图谱和 Milvus collection 可用，但不会主动连接 DeepSeek 或 Neo4j。Neo4j 查询失败时仍会降级到已经通过检查的本地 JSON 图谱。

## 测试

默认测试是完全离线的，不调用 LLM、Neo4j 或本地大模型：

```bash
python -m pytest
```

验证当前 JSON 图谱的实体、关系、端点类型、重复边和孤立节点：

```bash
python kb/scripts/validate_graph.py
```

手工端到端烟雾测试会产生模型调用费用，并要求全部外部依赖可用：

```bash
python tests/smoke_test.py
```

`tests/validate_*.py` 是早期 Prompt 实验脚本，不属于当前默认回归测试。

## 知识库构建

新增或替换文档统一使用：

```bash
python kb/scripts/ingest_pipeline.py --pdf /absolute/path/新文档.pdf
```

命令默认执行 MinerU、VLM、切分、完整图谱重建、来源校验、Embedding、按文档替换 Milvus，并在配置 Neo4j 时同步更新。所有生成步骤先写入 `kb/.staging`；任一步失败都不会提交正式数据，文件和向量提交失败会恢复旧版本，Neo4j 使用单事务替换。成功的图谱抽取批次按内容哈希缓存在 `kb/.cache/graph`，网络中断后重跑不会重复抽取未变化的批次。

只验证而不提交：

```bash
python kb/scripts/ingest_pipeline.py --pdf /absolute/path/新文档.pdf --dry-run --keep-staging
```

已有标准 MinerU/VLM 产物时可跳过这两个耗时阶段：

```bash
python kb/scripts/ingest_pipeline.py \
  --pdf kb/processed/文档名.pdf \
  --doc-name 文档名 \
  --reuse-processed
```

`kb/scripts/` 中的低层阶段脚本仍保留用于调试：

1. `split_pdf.py`：裁剪原始 PDF。
2. `parse_pdf.py`：通过 MinerU 解析版面、公式、表格和图片。
3. `describe_images.py`：使用本地 VLM 将图片替换成可检索描述。
4. `chunk_text.py`：按标题与语义边界切分文本。
5. `embed_and_ingest.py`：生成全部向量并重建 Milvus Lite collection；正常新增文档不要直接使用。
6. `ingest_document.py --doc 文档名`：仅执行低层向量 upsert，并验证产物时效与 manifest 校验和。
7. `build_graph.py`：抽取实体关系；统一入库命令会以暂存模式调用。

MinerU 客户端需要按其当前官方说明单独安装，并通过 `MINERU_TOKEN` 提供凭据。

## 主要目录

```text
app/       FastAPI、教学 Agent、关联推荐、证据池、配置和数据层
frontend/  React + TypeScript + Vite 前端，通过 /api 代理联调后端
skills/    三个 Skill 的说明与调用入口
kb/        原始教材、处理中间产物、chunk、图谱和向量库
tests/     离线回归测试、手工烟雾测试和早期 Prompt 验证脚本
docs/      历史设计与 Prompt 文档，仅供设计演进参考
```
