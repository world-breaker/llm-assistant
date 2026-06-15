"""
rag_pipeline.py —— RAG 检索增强生成核心管线
==============================================
功能: 接收用户问题 → 向量检索 → 拼 Prompt → Qwen 生成回答
嵌入: BGE-small-zh（专用嵌入模型，检索准确率碾压通用LLM）

面试讲解要点:
  1. 为什么换BGE：通用LLM的隐藏状态不是为语义匹配优化的，区分度差
  2. 混合检索（语义 + 关键词）的设计理由
  3. 相似度阈值的作用（防止"硬编"）
  4. Prompt 模板的设计思路
"""

import os
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# 配置
# ============================================================
MODEL_PATH = os.path.join(
    os.environ["USERPROFILE"],
    ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2___5-3B-Instruct"
)
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 24MB，比Qwen轻250倍，检索准10倍
COLLECTION_NAME = "llm_knowledge"
SIMILARITY_THRESHOLD = 0.4  # BGE+余弦：相关0.5+，无关<0.2。0.4兜底不过滤弱相关
DEFAULT_TOP_K = 4


# ============================================================
# Part 1: 嵌入 —— BGE 专用嵌入模型
# ============================================================
class BGEEmbedder:
    """BGE 嵌入模型封装。面试时讲：为什么换掉 Qwen 嵌入。"""
    def __init__(self, model_name: str = BGE_MODEL_NAME):
        print(f"[嵌入] 加载 BGE 嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[嵌入] 向量维度: {self.dim}")

    def encode(self, texts: list, is_query: bool = False) -> np.ndarray:
        """文本 → 向量。BGE 查询需加指令前缀，文档不需要。"""
        if isinstance(texts, str):
            texts = [texts]
        if is_query:
            # BGE 模型要求：查询文本前必须加这个指令前缀
            texts = ["为这个句子生成表示以用于检索相关文章：" + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)


# ============================================================
# Part 2: 混合检索 —— 语义 + 关键词
# ============================================================
SYSTEM_PROMPT = """你是一个专业的LLM技术面试辅导助手。你的知识涵盖Transformer架构、大模型训练与微调、RAG、RLHF/DPO、模型部署优化等方向。

回答规则:
1. 只根据「参考资料」中的信息回答，不要编造
2. 如果参考资料中没有相关信息，明确说"我不确定"并给出你的知识范围内的推测
3. 回答要准确、简洁，先给出核心结论再展开细节
4. 适当使用公式和技术术语，体现专业性"""


def keyword_score(query: str, document: str) -> float:
    """关键词匹配分数（Jaccard 相似度）"""
    query_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', query.lower()))
    doc_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', document.lower()))
    if not query_words:
        return 0.0
    intersection = query_words & doc_words
    union = query_words | doc_words
    return len(intersection) / len(union) if union else 0.0


def hybrid_search(
    query: str,
    collection,
    embedder: "BGEEmbedder",
    top_k: int = DEFAULT_TOP_K,
    semantic_weight: float = 0.7,
):
    """混合检索：BGE语义 × 0.7 + 关键词 × 0.3"""
    query_emb = embedder.encode([query], is_query=True)
    candidate_k = min(top_k * 3, collection.count())

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=candidate_k
    )

    scored_docs = []
    for i in range(len(results["ids"][0])):
        doc_text = results["documents"][0][i]
        semantic_sim = 1.0 - results["distances"][0][i]
        kw_score = keyword_score(query, doc_text)
        combined = semantic_weight * semantic_sim + (1 - semantic_weight) * kw_score

        scored_docs.append({
            "id": results["ids"][0][i],
            "content": doc_text,
            "source": results["metadatas"][0][i].get("source", "未知"),
            "semantic_score": round(semantic_sim, 3),
            "keyword_score": round(kw_score, 3),
            "combined_score": round(combined, 3)
        })

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
        self.project_root = project_root
        self.db_path = os.path.join(project_root, "chroma_db")

        # ── 加载 BGE 嵌入模型（轻量，先加载）──
        self.embedder = BGEEmbedder()

        # ── 加载 Qwen 生成模型（重量级）──
        print("[RAG] 加载 Qwen2.5-3B...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print(f"[RAG]   CUDA 可用: {torch.cuda.is_available()}")
        print(f"[RAG]   显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16,
        )
        self.model = self.model.cuda()

        if lora_path and os.path.exists(lora_path):
            print(f"[RAG] 加载 LoRA 权重: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

        device = next(self.model.parameters()).device
        print(f"[RAG] 生成模型设备: {device}")
        if device.type == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[RAG] 显存: {mem:.1f}GB / {total:.1f}GB")

        # 连接向量库
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"[RAG] 向量库就绪: {self.collection.count()} 条文档")
        except Exception:
            print("[RAG] 警告: 向量库未找到，请先运行 build_kb.py")
            self.collection = None

    def retrieve(self, query: str) -> list:
        """仅检索，不生成。返回匹配的文档列表。"""
        if not self.collection:
            return []
        retrieved = hybrid_search(
            query, self.collection, self.embedder, top_k=DEFAULT_TOP_K
        )
        return [d for d in retrieved if d["combined_score"] >= SIMILARITY_THRESHOLD]

    def ask_with_messages(
        self, messages: list, retrieved_docs: list = None, stream: bool = False
    ) -> dict:
        """
        用预构建的消息列表生成回答（支持多轮记忆）。
        如果有检索文档，拼入最后一条 user 消息前。
        """
        if retrieved_docs and len(retrieved_docs) > 0:
            doc_text = "\n\n".join(
                f"[参考{d['source']}] {d['content'][:300]}" for d in retrieved_docs[:3]
            )
            # 在最后一条 user 消息前插入检索结果
            enhanced = messages[-1]["content"]
            messages[-1]["content"] = f"参考资料:\n{doc_text}\n\n用户问题: {enhanced}\n\n请基于参考资料回答。"

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("assistant")[-1].strip() if "assistant" in response else response

        return {
            "retrieved_docs": [
                {"source": d["source"], "score": f"{d['combined_score']:.3f}", "preview": d["content"][:80]}
                for d in (retrieved_docs or [])
            ],
            "answer": answer,
        }

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
                query, self.collection, self.embedder, top_k=DEFAULT_TOP_K
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
