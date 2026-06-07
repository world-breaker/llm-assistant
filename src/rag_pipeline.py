"""
rag_pipeline.py —— RAG 检索增强生成核心管线
==============================================
功能: 接收用户问题 → 向量检索 → 拼 Prompt → Qwen 生成回答

面试讲解要点:
  1. 混合检索（语义 + 关键词）的设计理由
  2. 相似度阈值的作用（防止"硬编"）
  3. Prompt 模板的设计思路
  4. 生成参数（temperature/top_p/top_k）的选择
"""

import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import chromadb


# ============================================================
# 配置
# ============================================================
MODEL_PATH = os.path.join(
    os.environ["USERPROFILE"],
    ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2___5-3B-Instruct"
)
COLLECTION_NAME = "llm_knowledge"
SIMILARITY_THRESHOLD = 0.35  # Qwen嵌入的余弦相似度偏低，0.35即可过滤无关内容
DEFAULT_TOP_K = 4            # 检索返回文档数

# Prompt 模板 —— 面试时可以解释每一部分的设计意图
SYSTEM_PROMPT = """你是一个专业的LLM技术面试辅导助手。你的知识涵盖Transformer架构、大模型训练与微调、RAG、RLHF/DPO、模型部署优化等方向。

回答规则:
1. 只根据「参考资料」中的信息回答，不要编造
2. 如果参考资料中没有相关信息，明确说"我不确定"并给出你的知识范围内的推测
3. 回答要准确、简洁，先给出核心结论再展开细节
4. 适当使用公式和技术术语，体现专业性"""


# ============================================================
# Part 1: 嵌入 —— 复用 build_kb.py 的逻辑
# ============================================================
def encode_texts(texts: list, tokenizer, model) -> torch.Tensor:
    """用 Qwen 隐藏状态做句向量（与 build_kb.py 一致）"""
    embeddings = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            masked = last_hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1)
            embedding = summed / counts
            embedding = F.normalize(embedding, p=2, dim=-1)

        embeddings.append(embedding.cpu())
    return torch.cat(embeddings, dim=0)


# ============================================================
# Part 2: 混合检索 —— 语义 + 关键词
# ============================================================
def keyword_score(query: str, document: str) -> float:
    """
    关键词匹配分数（BM25 简化版）
    面试追问: "为什么加关键词匹配？"
    答: 纯语义检索会漏掉精确术语匹配。
       比如搜"GQA"，语义上不一定能关联到"分组查询注意力"的文档块，
       但关键词可以直接命中。两者互补。
    """
    # 提取中文词（2字以上）+ 英文单词
    query_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', query.lower()))
    doc_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', document.lower()))

    if not query_words:
        return 0.0

    # Jaccard 相似度：交集 / 并集
    intersection = query_words & doc_words
    union = query_words | doc_words
    return len(intersection) / len(union) if union else 0.0


def hybrid_search(
    query: str,
    collection,
    tokenizer,
    model,
    top_k: int = DEFAULT_TOP_K,
    semantic_weight: float = 0.7  # 语义权重：语义70%，关键词30%
):
    """
    混合检索：语义相似度 × 0.7 + 关键词匹配 × 0.3
    面试讲述: "先用向量库做语义召回 Top-2K，再在候选集上做关键词重排序"
    """
    # Step 1: 语义检索（先召回候选集）
    query_emb = encode_texts([query], tokenizer, model)
    candidate_k = top_k * 3  # 多召回一些候选，给关键词留重排空间

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=candidate_k
    )

    # Step 2: 关键词重排序
    scored_docs = []
    for i in range(len(results["ids"][0])):
        doc_text = results["documents"][0][i]
        semantic_sim = 1.0 - results["distances"][0][i]  # ChromaDB 的距离 → 相似度
        kw_score = keyword_score(query, doc_text)

        # 混合分数
        combined = semantic_weight * semantic_sim + (1 - semantic_weight) * kw_score

        scored_docs.append({
            "id": results["ids"][0][i],
            "content": doc_text,
            "source": results["metadatas"][0][i].get("source", "未知"),
            "semantic_score": semantic_sim,
            "keyword_score": kw_score,
            "combined_score": combined
        })

    # Step 3: 按混合分数排序，取 top_k
    scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored_docs[:top_k]


# ============================================================
# Part 3: Prompt 构造
# ============================================================
def build_prompt(query: str, retrieved_docs: list) -> str:
    """
    构造 RAG Prompt
    面试追问: "这个 prompt 模板为什么这样设计？"
    答: 1) System Prompt 定义了角色和规则，防止跑偏
        2) 参考资料用 --- 分隔，清晰标注来源，方便debug
        3) 最后重复一遍"规则"，强化约束（模型对结尾的指令更敏感）
    """
    # 拼接检索到的文档
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        if doc["combined_score"] >= SIMILARITY_THRESHOLD:
            context_parts.append(
                f"[参考资料{i+1}] (来源: {doc['source']}, 匹配度: {doc['combined_score']:.2f})\n{doc['content']}"
            )

    if not context_parts:
        context = "（无相关参考资料）"
    else:
        context = "\n\n---\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

---
参考资料:
{context}
---

用户问题: {query}

请基于以上参考资料回答。如果资料中没有相关信息，请说明。"""
    return prompt


# ============================================================
# Part 4: 生成
# ============================================================
class RAGPipeline:
    """
    RAG 完整管线类
    面试价值: 封装成类 → 展示工程化思维，不是"脚本选手"
    """

    def __init__(self, project_root: str, lora_path: str = None):
        """
        Args:
            project_root: 项目根目录 (llm_interview_assistant/)
            lora_path: LoRA 权重路径（可选，用于加载微调后的模型）
        """
        self.project_root = project_root
        self.db_path = os.path.join(project_root, "chroma_db")

        # 加载模型 —— 强制上 GPU
        print("[RAG] 加载 Qwen2.5-3B...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print(f"[RAG]   CUDA 可用: {torch.cuda.is_available()}")
        print(f"[RAG]   显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        print(f"[RAG]   显存空闲: {torch.cuda.memory_reserved(0)/1024**3:.1f}GB")

        # 先清理显存碎片
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
        )
        self.model = self.model.cuda()  # 强制移到 GPU

        # 加载 LoRA（如果有）
        if lora_path and os.path.exists(lora_path):
            print(f"[RAG] 加载 LoRA 权重: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()  # 合并权重，加速推理

        self.model.eval()

        # GPU 检测 —— 确认模型在显卡上而非 CPU
        device = next(self.model.parameters()).device
        print(f"[RAG] 模型设备: {device}")
        if device.type == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[RAG] 显存: {mem:.1f}GB / {total:.1f}GB")
        else:
            print("[RAG] ⚠️  警告: 模型不在 GPU 上！生成会非常慢！")

        # 连接向量库
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"[RAG] 向量库就绪: {self.collection.count()} 条文档")
        except Exception:
            print("[RAG] 警告: 向量库未找到，请先运行 build_kb.py")
            self.collection = None

    def ask(self, query: str, stream: bool = True) -> dict:
        """
        问一个问题，返回 RAG 增强后的回答
        终端打印每步耗时，便于诊断瓶颈
        """
        import time as _time
        _t_total = _time.time()

        # ── Step 1: 检索 ──
        _t = _time.time()
        retrieved = []
        if self.collection:
            print(f"  [检索] embedding...", end="", flush=True)
            retrieved = hybrid_search(
                query, self.collection, self.tokenizer, self.model, top_k=DEFAULT_TOP_K
            )
            print(f" {_time.time()-_t:.1f}s", flush=True)
        else:
            retrieved = []

        # ── Step 2: 相似度判断 ──
        best_score = retrieved[0]["combined_score"] if retrieved else 0
        if best_score < SIMILARITY_THRESHOLD:
            print(f"  [跳过] 无高相关文档 (最高 {best_score:.2f} < {SIMILARITY_THRESHOLD})")
            print(f"  [生成] 直接回答中...", end="", flush=True)
            # 无检索结果时也用模型直接回答
            _tg = _time.time()
            messages = [{"role": "user", "content": query}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,  # 贪心解码，速度最快
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("assistant")[-1].strip() if "assistant" in response else response
            print(f" {_time.time()-_tg:.1f}s", flush=True)
            print(f"  [总耗时] {_time.time()-_t_total:.1f}s", flush=True)
            return {
                "query": query,
                "retrieved_docs": [],
                "answer": answer,
                "prompt": query
            }

        # ── Step 3: 构造 Prompt ──
        print(f"  [构造] prompt...", end="", flush=True)
        _t = _time.time()
        prompt = build_prompt(query, retrieved)
        print(f" {_time.time()-_t:.1f}s ({len(prompt)}字)", flush=True)

        # ── Step 4: 生成 ──
        print(f"  [生成] 推理中...", end="", flush=True)
        _t = _time.time()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": 150,
            "do_sample": False,  # 贪心解码，快5-10倍
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        print(f" {_time.time()-_t:.1f}s", flush=True)

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("assistant")[-1].strip() if "assistant" in full_response else full_response

        print(f"  [总耗时] {_time.time()-_t_total:.1f}s", flush=True)

        return {
            "query": query,
            "retrieved_docs": [
                {
                    "source": d["source"],
                    "score": f"{d['combined_score']:.3f}",
                    "preview": d["content"][:80]
                }
                for d in retrieved if d["combined_score"] >= SIMILARITY_THRESHOLD
            ],
            "answer": answer,
            "prompt": prompt
        }


# ============================================================
# Part 5: 命令行测试
# ============================================================
if __name__ == "__main__":
    # 找到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 可选：指定 LoRA 路径
    lora_path = os.path.join(project_root, "models", "lora_checkpoint")

    rag = RAGPipeline(project_root, lora_path=None)

    # 测试问题
    test_queries = [
        "什么是LoRA？为什么它这么流行？",
        "RAG的工作流程是怎样的？",
        "量子计算对大模型有什么影响？",  # 知识库中没有的
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}")

        result = rag.ask(q, stream=True)

        print(f"\n检索到的文档:")
        for doc in result["retrieved_docs"]:
            print(f"  [{doc['score']}] {doc['source']}: {doc['preview']}...")
