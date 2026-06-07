"""
app.py —— LLM 技术面试智能助手（流式版）
==========================================
每次提问逐句出结果，不用等全部生成完
"""
import os, sys, time
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import RAGPipeline

rag = None

EXAMPLES = [
    "什么是Transformer？",
    "LoRA的原理是什么？",
    "RAG的工作流程是怎样的？",
]


def answer_question(message: str, history: list):
    """生成器：逐步返回内容，网页实时更新"""
    global rag
    if not message or not message.strip():
        yield "请输入问题"
        return
    if rag is None:
        yield "⏳ 模型加载中..."
        return

    print(f"\n[Q] {message[:60]}")
    t0 = time.time()

    # 执行 RAG
    try:
        result = rag.ask(message, stream=False)
    except Exception as e:
        import traceback; traceback.print_exc()
        yield f"❌ {e}"
        return

    # 逐段 yield，模拟流式
    answer = result.get("answer", "")
    docs = result.get("retrieved_docs", [])

    # 先出参考文档
    if docs:
        ref = "### 📚 参考文档\n"
        for d in docs:
            ref += f"- [{d['score']}] *{d['source']}*\n"
        ref += "\n---\n"
        yield ref + "⏳ 思考中..."
        time.sleep(0.3)

    # 逐段出回答
    full = ""
    if docs:
        full += "### 💡 回答\n"
    paragraphs = answer.split("\n\n")
    for i, para in enumerate(paragraphs):
        if para.strip():
            full += para
            if i < len(paragraphs) - 1:
                full += "\n\n"
            yield full

    print(f"  [OK] {time.time()-t0:.1f}s")


def main():
    global rag

    print("\n" + "=" * 60)
    print("🚀 LLM 技术面试智能助手")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 加载模型（带 LoRA 微调权重）
    print("[1/3] 加载 Qwen2.5-3B + LoRA 微调模型...")
    t0 = time.time()
    lora_path = os.path.join(project_root, "models", "lora_checkpoint")
    rag = RAGPipeline(project_root, lora_path=lora_path)
    print(f"      ✓ ({time.time()-t0:.0f}s)")

    # 测试生成
    print("\n[2/3] 测试生成...")
    test_msg = "你好，一句话介绍自己"
    messages = [{"role": "user", "content": test_msg}]
    text = rag.tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = rag.tokenizer(text, return_tensors="pt").to(rag.model.device)
    with torch.no_grad():
        outputs = rag.model.generate(
            **inputs, max_new_tokens=40, do_sample=False,
            pad_token_id=rag.tokenizer.eos_token_id,
        )
    resp = rag.tokenizer.decode(outputs[0], skip_special_tokens=True)
    ans = resp.split("assistant")[-1].strip() if "assistant" in resp else resp
    print(f"      回答: {ans[:80]}")
    print(f"      ✓ 测试通过")

    # 开网页
    print(f"\n[3/3] 打开浏览器 → http://127.0.0.1:7860")
    print("=" * 60 + "\n")

    demo = gr.ChatInterface(
        fn=answer_question,
        title="LLM 技术面试智能助手",
        description="Qwen2.5-3B + RAG | 盯终端看进度",
        examples=EXAMPLES,
    )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
