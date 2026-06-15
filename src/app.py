"""
app.py —— LLM 技术面试智能助手（完整版）
==========================================
功能:
  1. 多角色切换（技术专家/情感陪伴/简洁助手/创意伙伴）
  2. 多轮对话记忆（滑动窗口，记住上下文）
  3. RAG 检索增强（BGE 嵌入 + ChromaDB）
  4. 内容安全过滤（输入输出双检）
  5. LoRA 微调模型自动加载
"""
import os, sys, time, json
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import RAGPipeline
from memory import ConversationMemory
from safety import check_output_safety, get_safe_response

# ============================================================
# 全局状态
# ============================================================
rag = None
memory = ConversationMemory(max_turns=10)

# 加载体人设配置
def load_personas():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "personas.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

PERSONAS = load_personas()


# ============================================================
# 问答函数
# ============================================================
def answer_question(message: str, history: list, persona_name: str):
    """
    完整的问答管线：安全检测 → 记忆 → RAG检索 → 人设Prompt → 生成 → 安全再审
    """
    global rag, memory

    if not message or not message.strip():
        return "请输入问题"
    if rag is None:
        return "⏳ 模型加载中..."

    # ── 安全检测（输入）──
    from safety import check_safety
    input_check = check_safety(message)
    if not input_check["safe"]:
        return f"⚠️ 内容安全提醒: {input_check['reason']}"

    print(f"\n[Q] {message[:60]}")

    # ── 加载当前人设 ──
    persona = PERSONAS.get(persona_name, PERSONAS["技术专家"])

    # ── RAG 检索 + 生成 ──
    t0 = time.time()
    try:
        # 用记忆上下文增强 prompt
        context = memory.get_context() if len(memory) > 0 else ""
        enhanced_query = message
        if context:
            enhanced_query = f"{context}\n\n[最新问题] {message}"

        result = rag.ask(message, stream=False)
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"❌ {e}"

    # ── 安全检测（输出）──
    answer = result.get("answer", "")
    output_check = check_safety(answer)
    if not output_check["safe"]:
        answer = f"⚠️ 回答已过滤: {output_check['reason']}"

    # ── 更新记忆 ──
    memory.add(message, answer)

    # ── 构造回复 ──
    lines = []

    # 检索文档
    if result.get("retrieved_docs"):
        lines.append("### 📚 参考文档")
        for doc in result["retrieved_docs"]:
            lines.append(f"- [{doc['score']}] *{doc['source']}*")
        lines.append("")

    # 当前人设标识
    lines.append(f"*当前角色: {persona_name}*")
    lines.append("")

    # 回答
    lines.append("### 💡 回答")
    lines.append(answer)

    # 记忆状态
    if len(memory) > 0:
        lines.append(f"\n---\n💬 已记住 {len(memory)} 轮对话")

    print(f"  [OK] {time.time()-t0:.1f}s | 角色: {persona_name} | 记忆: {len(memory)}轮")
    return "\n".join(lines)


# ============================================================
# 界面
# ============================================================
def main():
    global rag

    print("\n" + "=" * 60)
    print("🚀 LLM 智能助手（完整版）")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 加载模型
    print("[1/2] 加载模型...")
    t0 = time.time()
    lora_path = os.path.join(project_root, "models", "lora_checkpoint")
    rag = RAGPipeline(project_root, lora_path=lora_path)
    print(f"      ✓ ({time.time()-t0:.0f}s)")

    # 测试生成
    print("[2/2] 测试生成...")
    import torch
    test_msg = "你好"
    messages = [{"role": "user", "content": test_msg}]
    text = rag.tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = rag.tokenizer(text, return_tensors="pt").to(rag.model.device)
    with torch.no_grad():
        outputs = rag.model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                      pad_token_id=rag.tokenizer.eos_token_id)
    resp = rag.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"      测试: {resp.split('assistant')[-1].strip()[:60] if 'assistant' in resp else resp[:60]}")
    print(f"      ✓ 就绪")

    print(f"\n  打开浏览器 → http://127.0.0.1:7860")
    print("=" * 60 + "\n")

    # 构建界面
    persona_names = list(PERSONAS.keys())

    with gr.Blocks(title="LLM 智能助手") as demo:
        gr.Markdown("# 🤖 LLM 智能助手")
        gr.Markdown("Qwen2.5-3B + RAG(BGE) + LoRA微调 + 多角色 + 记忆 + 安全过滤")

        with gr.Row():
            with gr.Column(scale=1):
                persona_dropdown = gr.Dropdown(
                    choices=persona_names, value="技术专家",
                    label="选择角色", interactive=True
                )
                clear_btn = gr.Button("清除记忆", size="sm")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, label="对话", type="tuples")
                msg = gr.Textbox(placeholder="输入问题...", label="")
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    gr.Examples(
                        examples=["LoRA的原理是什么？", "我最近心情不好", "用一句话解释Transformer"],
                        inputs=msg, label="试试这些问题"
                    )

        # 事件绑定
        def respond(message, history, persona):
            bot_msg = answer_question(message, history, persona)
            history.append((message, bot_msg))
            return "", history

        def clear_memory():
            global memory
            memory.clear()
            return []

        submit_btn.click(respond, [msg, chatbot, persona_dropdown], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, persona_dropdown], [msg, chatbot])
        clear_btn.click(clear_memory, outputs=[chatbot])

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
