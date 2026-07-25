"""
向量嵌入生成 & Milvus 入库（本地 Qwen3-Embedding-0.6B）

═══════════════════════════════════════════════════════════════
什么是 Embedding？
  把一段文字变成一个浮点数数组（向量）。语义相近的文字，向量在空间中
  方向相近（余弦相似度接近 1）；语义无关的文字，向量方向接近正交（≈0）。

什么是 Milvus？
  一个专门存向量 + 做相似度搜索的数据库。你给它一个查询向量，它用近似
  最近邻（ANN）算法快速找到最相似的 K 个向量，返回对应的原始数据。

本脚本做了什么？
  1. 读取 data/chunks_*.jsonl 中的每个 chunk
  2. 用 Qwen3-Embedding-0.6B 把 chunk["text"] 转成 1024 维向量
  3. 把向量 + 原文 + 元数据写入 Milvus Lite（嵌入式，无需 Docker）
  4. 后续检索时，只需把用户问题转成向量 → 查 Milvus → 返回最相关 chunk

输出: vectordb/milvus.db  (Milvus Lite 数据文件)
用法: python build_kb/embed_and_ingest.py
═══════════════════════════════════════════════════════════════
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# pymilvus: Milvus 的 Python 客户端
#   MilvusClient:  连接到 Milvus 数据库（支持 Lite 嵌入模式 / Server 远程模式）
#   DataType:      定义字段类型（VARCHAR / FLOAT_VECTOR / INT16 等）
#   CollectionSchema:  定义"表"的结构（有哪些字段，每个字段什么类型）
#   FieldSchema:       定义"列"的结构（字段名、类型、是否主键等）
from pymilvus import MilvusClient, DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema

# SentenceTransformer: 一行代码加载 Embedding 模型，自动处理：
#   1. tokenizer:    把文字切成 token 并映射为数字 ID
#   2. 模型前向传播:   把 token ID 序列传入 Transformer，得到隐藏状态
#   3. pooling:      把最后一层所有 token 的隐藏状态聚合成一个向量
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

# __file__ 是当前脚本的路径（如 kb/scripts/embed_and_ingest.py）
# .resolve() 转绝对路径，.parent 取其所在目录，再 .parent.parent 取项目根
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# 输入目录：存放 chunk_text.py 生成的 JSONL 文件
CHUNKS_DIR = BASE_DIR / "kb" / "data"

# 输出目录：向量库的持久化文件放在这里
VECTORDB_DIR = BASE_DIR / "kb" / "vectordb"
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

# Milvus Lite 用本地文件存储所有数据，不需要启动独立服务
# str() 转字符串是因为后续 pymilvus 需要字符串路径而非 Path 对象
DB_FILE = str(VECTORDB_DIR / "milvus.db")

# Collection ≈ 关系数据库中的"表"
# 一个 collection 里存了 id、向量、文本等字段的所有数据
# 这里取名 "linear_algebra_kb" 因为只存了《线性表》这一本教材
# 后续如果要存多本教材，可以用同一个 collection（靠 doc_name 字段区分）
# 也可以每本教材一个 collection（隔离更好，但查询时需要跨表搜索）
COLLECTION_NAME = "linear_algebra_kb"

# ═══════════════════════════════════════════════════════════════
# Embedding 模型配置
# ═══════════════════════════════════════════════════════════════

# Qwen3-Embedding-0.6B:
#   - 600M 参数（0.6B），比 4B 版小约 7 倍，CPU 上快约 8 倍
#   - 输出 1024 维向量（4B 版是 2560 维）
#   - 最大输入 32K token，远超每个 chunk 的 500 字
#   - 本地推理，首次加载约 30 秒，之后常驻内存
#   - 文件约 1.2GB（fp16），远小于 4B 版的 8GB
MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    str(BASE_DIR.parent / "models" / "Qwen" / "Qwen3-Embedding-0.6B"),
)

# Instruction Tuning 的查询前缀
#   为什么只加在查询侧？
#     Qwen3-Embedding 训练时用了 instruction tuning：给 query 加指令前缀，
#     document 不加。这样模型知道"现在要编码查询"，会把查询向量映射到
#     与文档向量对齐的同一语义空间中更精确的位置。
#   这个前缀来自模型目录下的 config_sentence_transformers.json 中的 prompts.query
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

# 批处理大小（一次往模型送多少条文本）
#   - 值越大 GPU 利用率越高，但 CPU 上差距不大（CPU 本身就是串行处理）
#   - 值越小内存占用越低，但循环次数多开销也大
#   - 16 是一个中值，128 条数据跑 8 批
#   - 写入 Milvus 时也用这个值做分批，避免一次写太多导致超时
BATCH_SIZE = 16


# ═══════════════════════════════════════════════════════════════
# Schema 定义 —— 决定 Milvus 中存什么、怎么存
# ═══════════════════════════════════════════════════════════════

def build_schema(dim: int):
    """
    构建 Milvus Collection 的 Schema（表结构定义）。

    Args:
        dim: 向量维度。由模型自动决定，传入 model.get_embedding_dimension() 的返回值。
             0.6B 模型 → 1024，4B 模型 → 2560

    Returns:
        CollectionSchema 对象

    Milvus 字段类型对照:
      VARCHAR     → 变长字符串，需指定 max_length（字符数上限）
      FLOAT_VECTOR→ 浮点数数组，需指定 dim（数组长度 = 向量维度）
      INT16       → 16 位整数（-32768 ~ 32767），适合页码、字符数

    字段详解:
      id          — chunk 唯一标识（如 "chunk_001"）。is_primary=True 表示主键，
                    重复插入同 ID 会覆盖旧数据
      vector      — 语义向量（1024 个 float32）。这是检索的核心：
                    Milvus 对查询向量和所有文档向量做余弦相似度，排序取 TopK
      text        — chunk 原文（最多 4096 字符）。检索命中后直接取这个字段拼入 LLM
                    prompt，不需要再回文件系统读
      header_path — 标题路径（如 "## 顺序表的插入 > ## 3) 时间复杂度"）。
                    回答时展示给用户看，表示这段内容来自教材哪个位置
      doc_name    — 来源文档名（如 "线性表"）。多文档入库后可以：
                    expr="doc_name=='线性表'" 限定只搜某一本书
      page_num    — 页码。用户点击溯源时，用这个数字跳转到 PDF 对应页
      char_count  — 字符数。调试时用：如果某个 chunk 太小或太大，可以快速定位
    """
    # 用列表定义每个字段，顺序不重要，Milvus 会自动组织
    fields = [
        # 主键字段：必须唯一，用于去重和按 ID 删除
        # max_length=32 足够存 "chunk_001" 到 "chunk_999"
        FieldSchema("id",          DataType.VARCHAR, max_length=32,  is_primary=True),

        # 向量字段：dim 在高维空间中坐标的数量
        # dim 越大通常表达能力越强，但检索越慢、存储越大
        # 这里 dim=1024，每条向量 1024×4=4096 字节 ≈ 4KB
        FieldSchema("vector",      DataType.FLOAT_VECTOR, dim=dim),

        # 原文字段：max_length=4096 覆盖 chunk_size=500 + 一些 buffer
        # 即使未来调大 chunk_size 到 2000 字也够用
        FieldSchema("text",        DataType.VARCHAR, max_length=4096),

        # 以下都是元数据字段，检索时不参与计算，但结果展示和过滤需要
        FieldSchema("header_path", DataType.VARCHAR, max_length=512),
        FieldSchema("doc_name",    DataType.VARCHAR, max_length=128),
        FieldSchema("page_num",    DataType.INT16),
        FieldSchema("char_count",  DataType.INT16),
    ]

    # 把字段列表包装成 Schema 对象
    # description 是可选的注释，方便在 Milvus 管理界面中识别
    return CollectionSchema(fields, description="线性表知识库 chunks")


# ═══════════════════════════════════════════════════════════════
# 索引配置
# ═══════════════════════════════════════════════════════════════

def build_index_params(client: MilvusClient):
    """
    为向量字段创建索引。没有索引的话检索只能暴力扫描（O(N)），
    有索引后用近似最近邻（ANN），速度提升几个数量级。

    IVF_FLAT 索引原理（两步检索）:
      第一步：用 K-Means 把全部向量聚成 nlist 个簇（nlist=16 → 16 个中心点）
      第二步：查询时先把查询向量和 16 个中心比较，选最近的 nprobe 个簇
             只在这几个簇内暴力扫描找 TopK
      代价：可能漏掉真正最近的那个簇 → 近似而非精确

    为什么用 COSINE 距离？
      Qwen3-Embedding 训练时把相似函数设为了 cosine，推理也保持一致性。
      而且 normalize_embeddings=True 后向量已经 L2 归一化（长度为 1），
      此时 COSINE(a,b) = a·b（向量内积），计算最快。

    nlist 怎么选？
      经验公式 nlist = 4 × √N。N=128 → √128≈11 → nlist≈45
      但 nlist 不能超过 N（128 条分 45 个簇平均每簇不到 3 条，没有意义）
      这里设 nlist=16，使每个簇平均 128/16 = 8 条，检索时探 2-4 个簇即可
    """
    # prepare_index_params 返回 IndexParams 对象（还没绑定任何索引配置）
    index_params = client.prepare_index_params()

    # add_index: 给指定字段加一个索引
    #   field_name:  哪个字段需要索引（只有向量字段需要，标量字段不需要）
    #   index_type:  索引算法类型
    #     IVF_FLAT      — 聚类 + 平坦扫描（最常用，内存适中，召回率高）
    #     IVF_SQ8       — 聚类 + 8bit 量化（省内存但损失精度）
    #     HNSW          — 分层可导航小世界图（最快但内存占用大）
    #   metric_type: 距离度量方式
    #     COSINE  — 余弦相似度（1=完全同向, 0=正交, -1=完全反向）
    #     IP      — 内积（向量已归一化时等价于 COSINE）
    #     L2      — 欧几里得距离（越小越相似，常用于未归一化向量）
    #   params: 索引算法特定参数
    #     nlist  — IVF 的聚类中心数
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",       # 小数据集用 FLAT（暴力扫描），当前规模无需聚类
        metric_type="COSINE",
    )

    return index_params


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("向量嵌入 & Milvus 入库  (Qwen3-Embedding-0.6B)")
    print("=" * 60)

    # ── 1. 加载 Embedding 模型 ─────────────────────────────────
    # SentenceTransformer 是 sentence-transformers 的核心类。
    # 构造函数做的事：
    #   a) 从 MODEL_PATH 读取 config.json，知道模型架构是 Qwen3ForCausalLM
    #   b) 读取 config_sentence_transformers.json，知道 pooling 方式和 query prompt
    #   c) 加载 safetensors 权重文件（约 1.2GB）到内存
    #   d) 自动选择 pooling 策略（Qwen3 用最后一层 mean pooling + 归一化层）
    #
    # device="cpu": 用 CPU 推理。Apple MPS 可能不稳定，保守用 CPU。
    #   0.6B 模型 CPU 上 128 条约 6 分钟，可接受。
    print(f"\n⏳ 加载模型: {MODEL_PATH}")
    model = SentenceTransformer(
        MODEL_PATH,
        device="cpu",
    )
    # 自动获取模型输出的向量维度，无需手动填写
    dim = model.get_embedding_dimension()
    print(f"   ✅ 加载完成  |  向量维度: {dim}  |  设备: {model.device}")

    # ── 2. 连接 Milvus Lite ─────────────────────────────────
    #
    # MilvusClient(uri): uri 决定了连接模式
    #   "./vectordb/milvus.db"  → 本地文件路径 → Milvus Lite 嵌入式模式
    #   "http://localhost:19530" → HTTP URL     → Milvus 独立服务模式
    #
    # Lite 模式的特点：
    #   - 数据存在本地 SQLite 文件 + 向量索引文件
    #   - 不需要安装 Docker 或启动守护进程
    #   - 适合开发、小型项目（< 百万条数据）
    #   - 查询时进程内直接操作文件，网络延迟为零
    print(f"\n⏳ 连接 Milvus Lite: {DB_FILE}")
    client = MilvusClient(DB_FILE)

    # ── 3. 创建 Collection ────────────────────────────────────
    # has_collection: 检查集合是否已存在（避免重复创建报错）
    # drop_collection: 删除集合及其所有数据（为了 schema 一致性重建）
    #   正式环境一般用 alias（别名）做无宕机迁移，而非删库重建
    if client.has_collection(COLLECTION_NAME):
        print(f"   ⚠️  Collection 已存在，删除重建...")
        client.drop_collection(COLLECTION_NAME)

    schema = build_schema(dim)
    index_params = build_index_params(client)

    # create_collection: 同时指定 schema 和 index_params
    # Milvus 会在创建时自动建索引（也可以先建表再后期加索引）
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"   ✅ Collection '{COLLECTION_NAME}' 创建完成")

    # ── 4. 遍历 JSONL → Embedding → 入库 ─────────────────────
    # glob("chunks_*.jsonl"): 通配查找所有 JSONL 文件
    #   sorted(): 保证处理顺序一致（按文件名排序）
    jsonl_files = sorted(CHUNKS_DIR.glob("chunks_*.jsonl"))
    if not jsonl_files:
        print("❌ 未找到 chunks_*.jsonl，请先运行 chunk_text.py")
        return

    total_inserted = 0  # 累计入库条数

    for jsonl_path in jsonl_files:
        # 从文件名提取文档名
        #   jsonl_path.stem = "chunks_线性表"（去掉 .jsonl 后缀）
        #   .replace("chunks_", "") = "线性表"（去掉前缀）
        doc_name = jsonl_path.stem.replace("chunks_", "")
        print(f"\n📄 {jsonl_path.name}  →  文档: {doc_name}")

        # 4a. 读取所有 chunk
        #     JSONL 格式：每行一个完整的 JSON 对象
        #     不像 JSON 数组需要一次加载整个文件，JSONL 可以逐行流式读取
        chunks = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    chunks.append(json.loads(line))
        print(f"   Chunk 数: {len(chunks)}")

        # 4b. 提取纯文本列表
        #     列表推导式：[c["text"] for c in chunks]
        #     得到 ["# 第2章...", "## 1. 总体要求...", ...] 这样的列表
        texts = [c["text"] for c in chunks]

        # 4c. 生成向量（核心步骤）
        #     model.encode() 的完整流程：
        #       1. tokenize: 把每段文字切分成 token，padding 到统一长度
        #       2. forward:  把 token 向量传入 36 层 Transformer，得到隐藏状态
        #       3. pooling:  取最后一层所有 token 的 hidden state 做 mean pooling
        #       4. normalize: L2 归一化（每个向量除以其模长），使所有向量长度为 1
        #
        #     参数说明:
        #       batch_size: 一次送入模型几条数据。CPU 上 batch 多大差别不大
        #       show_progress_bar: 显示 tqdm 进度条（剩余时间和速度）
        #       normalize_embeddings: 启用后向量长度为 1，余弦相似度 = 内积
        print(f"   🔄 生成向量  ({dim} 维, CPU)...")
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # 4d. 组装入库数据
        #     zip(chunks, embeddings): 把 chunk 和对应向量配对
        #     每个 chunk 生成一条记录，包含所有 schema 中定义的字段
        #
        #     embedding.tolist(): numpy 数组 → Python 列表
        #       numpy 数组不能直接存 Milvus，必须转成 list
        #
        #     chunk.get("page_num", 0): 安全取值
        #       当前 chunk 还没有 page_num 信息（默认 0），等后面改进
        insert_data = []
        for chunk, embedding in zip(chunks, embeddings):
            insert_data.append({
                "id":          chunk["id"],
                "vector":      embedding.tolist(),
                "text":        chunk["text"],
                "header_path": chunk.get("header_path", ""),
                "doc_name":    doc_name,
                "page_num":    chunk.get("page_num", 0),
                "char_count":  chunk.get("char_count", 0),
            })

        # 4e. 分批写入
        #     range(0, len(data), BATCH_SIZE): 生成 [0, 16, 32, ...]
        #     data[i : i+BATCH_SIZE]: 切片取当前批的数据
        #     为什么分批？一次性写 128 条可能内存或网络超时，分批更稳定
        print(f"   📥 写入 Milvus...")
        for i in range(0, len(insert_data), BATCH_SIZE):
            batch = insert_data[i : i + BATCH_SIZE]
            # insert 操作是幂等的：同 ID 重复插入会覆盖（因为 id 是主键）
            client.insert(collection_name=COLLECTION_NAME, data=batch)
        total_inserted += len(insert_data)
        print(f"   ✅ {len(insert_data)} 条已入库")

    # ── 5. 加载 Collection 到内存 ─────────────────────────────
    # load_collection: 把索引和原始数据加载到内存
    #   不调用这步的话后续检索会报错
    #   这一步会：1) 读取 IVF 索引结构  2) 缓存热点数据到内存
    #   对于 128 条数据几乎是瞬间完成
    client.load_collection(COLLECTION_NAME)

    # ── 6. 汇总 ───────────────────────────────────────────────
    # get_collection_stats: 返回集合级别统计
    #   row_count: 总行数（即 chunk 总数）
    #   还有 segments 等内部信息，这里只关心行数
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"\n{'='*60}")
    print(f"🎉 入库完成")
    print(f"   Collection:  {COLLECTION_NAME}")
    print(f"   总条数:      {stats['row_count']}")
    print(f"   向量维度:    {dim}")
    print(f"   存储位置:    {DB_FILE}")
    print(f"   下一阶段:    检索 & 问答")
    print("=" * 60)


# Python 标准写法：这个文件直接运行时执行 main()，被 import 时不执行
if __name__ == "__main__":
    main()
