"""
build_kb.py —— 构建LLM技术知识库
====================================
功能: 把 data/knowledge/ 下的技术文档分块、向量化、存入 ChromaDB
嵌入方式: 用 Qwen2.5-3B 的隐藏状态做句向量（无需额外模型）

面试讲解要点:
  1. 分块策略：为什么用 overlap、size 怎么选
  2. 为什么用 LLM 隐藏状态而非 sentence-transformers
  3. ChromaDB vs FAISS 的选择理由
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from chromadb.config import Settings
import glob

# ============================================================
# 配置参数（面试时可以解释每个参数的意义）
# ============================================================
CHUNK_SIZE = 600        # 每块最大字符数。600字足够覆盖一个技术概念
CHUNK_OVERLAP = 100     # 块间重叠字符数。避免关键信息被切断
COLLECTION_NAME = "llm_knowledge"  # ChromaDB 集合名
EMBEDDING_DIM = 2048    # Qwen2.5-3B 的隐藏层维度

# 模型路径（你 ModelScope 下载好的）
MODEL_PATH = os.path.join(
    os.environ["USERPROFILE"],
    ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2___5-3B-Instruct"
)

# ============================================================
# Part 1: 文档加载与分块
# ============================================================
def load_and_chunk(knowledge_dir: str):
    """
    加载所有 .md 文档，按 CHUNK_SIZE 分块，块间保留 overlap
    面试追问: "为什么 overlap 选100?"
    答: 块边界处可能出现关键句子被切断的情况。
       100字 overlap 约等于1-2个句子，足以覆盖切断点。
    """
    chunks = []
    sources = []  # 记录每块来自哪个文件（检索时可以溯源）

    md_files = glob.glob(os.path.join(knowledge_dir, "*.md"))
    print(f"[建库] 找到 {len(md_files)} 个知识文档")

    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # 滑动窗口分块
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            sources.append(filename)
            start += (CHUNK_SIZE - CHUNK_OVERLAP)  # 每次滑动 CHUNK_SIZE - OVERLAP

        print(f"  {filename}: {text[start:].count('') if start>=len(text) else (len(text)//(CHUNK_SIZE-CHUNK_OVERLAP) + 1)} 块")

    print(f"[建库] 共 {len(chunks)} 个文本块")
    return chunks, sources


# ============================================================
# Part 2: 嵌入模型 —— 用 Qwen 做句向量
# ============================================================
def load_embedder():
    """
    加载 Qwen2.5-3B，用于提取句向量
    面试追问: "为什么不用 sentence-transformers?"
    答: 1) Qwen 已经在硬盘上了，不需要额外下载
        2) Qwen 在大量中文语料上训练，中文语义理解更好
        3) 推理场景已经在用 Qwen，加载一份模型同时做 embed 和生成，省显存
    实际考量: 如果你的场景是高并发检索，专用 embed 模型（如 BGE）
             推理更快（几百MB vs 3B），但我们现在做 demo 不需要
    """
    print("[嵌入] 加载 Qwen2.5-3B 作为嵌入模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print(f"[嵌入] 模型就绪，隐藏层维度: {EMBEDDING_DIM}")
    return tokenizer, model


def encode_texts(texts: list, tokenizer, model) -> torch.Tensor:
    """
    用 LLM 隐藏状态做句向量
    原理: 文本 → Qwen → 取最后一层 hidden_states → 对 token 维度求平均
    面试追问: "为什么用平均池化而不是取 [CLS] token?"
    答: Qwen 没有 [CLS] token（那是 BERT 的设计）。
       平均池化是 decoder-only 模型做句子嵌入的标准做法。
    """
    embeddings = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, 2048)

            # 平均池化: 用 attention_mask 排除 padding
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
            masked = last_hidden * mask       # padding 位置置零
            summed = masked.sum(dim=1)        # 沿序列维度求和
            counts = mask.sum(dim=1)          # 每个样本的实际 token 数
            embedding = summed / counts       # 平均

            # L2 归一化 —— 让余弦相似度计算变成简单的内积
            embedding = F.normalize(embedding, p=2, dim=-1)

        embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)  # (num_texts, 2048)


# ============================================================
# Part 3: ChromaDB —— 向量存储与检索
# ============================================================
def build_chromadb(chunks: list, sources: list, embeddings: torch.Tensor, db_path: str):
    """
    将文本块和向量存入 ChromaDB
    面试追问: "为什么选 ChromaDB 而不是 FAISS?"
    答: FAISS 是纯粹的向量搜索引擎，不存原始文本。
       ChromaDB 自带文档存储 + 元数据管理 + 持久化，
       对小规模（<10万条）demo 来说开箱即用，不需要单独维护文本索引。
       如果数据量到百万级别，会考虑 Milvus 或 Elasticsearch。
    """
    print(f"[向量库] 存入 ChromaDB: {db_path}")

    client = chromadb.PersistentClient(path=db_path)

    # 删除旧集合（如果存在），确保重建
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "LLM技术面试知识库"}
    )

    # 批量插入
    embedding_list = embeddings.tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": src, "chunk_index": i} for i, src in enumerate(sources)]

    # ChromaDB 单次插入上限，分批处理
    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embedding_list[i:end],
            documents=chunks[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"  已插入 {end}/{len(chunks)} 条")

    print(f"[向量库] 完成！共 {collection.count()} 条文档")
    return collection


# ============================================================
# Part 4: 主流程
# ============================================================
def main():
    # 路径定义
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_dir = os.path.join(project_root, "data", "knowledge")
    db_path = os.path.join(project_root, "chroma_db")

    # 1. 加载文档并分块
    chunks, sources = load_and_chunk(knowledge_dir)

    # 2. 加载嵌入模型
    tokenizer, model = load_embedder()

    # 3. 向量化（分批处理，避免 OOM）
    print(f"[嵌入] 正在向量化 {len(chunks)} 个文本块...")
    BATCH = 8  # 每批8个，避免显存爆炸
    all_embs = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        embs = encode_texts(batch, tokenizer, model)
        all_embs.append(embs)
        print(f"  进度: {min(i+BATCH, len(chunks))}/{len(chunks)}")
    embeddings = torch.cat(all_embs, dim=0)
    print(f"[嵌入] 完成！向量形状: {embeddings.shape}")

    # 4. 存入 ChromaDB
    build_chromadb(chunks, sources, embeddings, db_path)

    # 5. 测试检索
    print("\n" + "=" * 50)
    print("[测试] 验证检索功能")
    print("=" * 50)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)

    test_query = "LoRA怎么节省显存？"
    query_emb = encode_texts([test_query], tokenizer, model)

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=3
    )

    print(f"\n查询: '{test_query}'")
    for i, (doc_id, doc_text, distance) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    )):
        print(f"\n  Top-{i+1} (距离={distance:.4f}):")
        print(f"  来源: {results['metadatas'][0][i]['source']}")
        print(f"  内容: {doc_text[:100]}...")

    print("\n✓ 知识库构建完成！")


if __name__ == "__main__":
    main()
