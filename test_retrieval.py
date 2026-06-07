"""测试检索是否工作"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import RAGPipeline

rag = RAGPipeline(r"D:\programmer\llm_interview_assistant")

# 用文档原句测试
tests = [
    "自注意力机制的核心公式",
    "LoRA通过低秩分解",
    "DPO直接用好坏回答对来优化",
    "KV Cache的原理",
]

for q in tests:
    result = rag.ask(q, stream=False)
    docs = result.get("retrieved_docs", [])
    best = result.get("retrieved_docs", [{}])[0].get("score", "N/A") if docs else "N/A"
    print(f"\n问题: {q}")
    print(f"  最高相似度: {best}")
    if docs:
        for d in docs:
            print(f"  [{d['score']}] {d['source']}")
    else:
        print(f"  ❌ 0条命中（阈值0.45）")
