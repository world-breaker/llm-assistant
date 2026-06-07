"""
sft_train.py —— LoRA 微调 Qwen2.5-3B
======================================
功能: 用 data/sft_train.json 中的技术问答对微调模型
     让模型在回答 LLM 技术问题时更准确、更有条理

面试讲解要点:
  1. 为什么选 LoRA 而不是全量微调（显存 + 实际需求）
  2. LoRA 的 r/alpha/target_modules 怎么选
  3. 训练数据怎么构造（数量、质量、多样性）
  4. 怎么评估微调效果（训练前后对比）
"""

import os
import json
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


# ============================================================
# 配置
# ============================================================
MODEL_PATH = os.path.join(
    os.environ["USERPROFILE"],
    ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2___5-3B-Instruct"
)

# LoRA 配置
# 面试追问: "r=8 怎么来的？"
# 答: r 越大表达能力越强但参数越多。经验值 r=4~16。
#     r=8 是社区验证的性价比最优值，适用于大多数场景。
#     alpha=16 意味着 LoRA 更新的实际强度 = alpha/r × BA = 2 × BA
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 层
        "gate_proj", "up_proj", "down_proj"         # FFN 层
    ],
}

# 训练配置
TRAINING_CONFIG = {
    "num_epochs": 5,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,  # 等效 batch_size = 4
    "learning_rate": 2e-4,
    "max_length": 512,
    "save_steps": 50,
}

# ============================================================
# Part 1: 加载数据
# ============================================================
def load_sft_data(data_path: str) -> Dataset:
    """
    加载 SFT 训练数据，格式化为对话模板
    面试追问: "18条数据够吗？"
    答: SFT 的核心是教会模型回答格式和风格，不是灌输知识。
       18条高质量、风格一致的问答对，足以让模型学会"技术问答"的语气和组织方式。
       知识获取靠的是预训练（万亿token）和 RAG，SFT 只是"调教"。
       当然如果追求最佳效果，建议200-500条。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"[数据] 加载 {len(raw_data)} 条训练样本")

    # 格式化为对话模板
    formatted_data = []
    for item in raw_data:
        messages = [
            {"role": "system", "content": "你是LLM技术专家，回答要准确、简洁、有深度。"},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]}
        ]
        formatted_data.append({"text": messages})

    dataset = Dataset.from_list(formatted_data)
    return dataset


# ============================================================
# Part 2: 数据预处理
# ============================================================
def preprocess_data(dataset: Dataset, tokenizer) -> Dataset:
    """
    将对话消息转成 token IDs
    面试追问: "apply_chat_template 做了什么？"
    答: 把 [{"role":"user","content":"你好"}] 转成
       "<|im_start|>user\n你好<|im_end|><|im_start|>assistant\n"
       这是 Qwen 训练时用的格式，错误的格式会导致模型完全不懂你在说什么。
    """
    def format_and_tokenize(examples):
        texts = []
        for msgs in examples["text"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=TRAINING_CONFIG["max_length"],
            padding=False
        )
        return tokenized

    dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
    return dataset


# ============================================================
# Part 3: 训练前后对比
# ============================================================
def test_before_after(model, tokenizer, test_prompts: list, tag: str):
    """训练前后各测一次，直观展示变化"""
    print(f"\n{'='*50}")
    print(f"[测试] {tag}")
    print(f"{'='*50}")

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 assistant 回答部分
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print(f"\n问题: {prompt}")
        print(f"回答: {response[:200]}")
        print("-" * 40)


# ============================================================
# Part 4: 主流程
# ============================================================
def main():
    # 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "sft_train.json")
    output_dir = os.path.join(project_root, "models", "lora_checkpoint")

    # ========================================
    # Step 1: 加载基座模型
    # ========================================
    print("=" * 50)
    print("[Step 1/6] 加载 Qwen2.5-3B")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ tokenizer 就绪")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"  ✓ 模型加载完成, 设备: {model.device}")

    # ========================================
    # Step 2: 配置 LoRA
    # ========================================
    print(f"\n[Step 2/6] 配置 LoRA (r={LORA_CONFIG['r']})")

    model = get_peft_model(model, LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    ))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total:,}")
    print(f"  可训练: {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"  这就是 LoRA 的核心: 只训练 {trainable/total*100:.1f}% 的参数")

    # ========================================
    # Step 3: 训练前测试
    # ========================================
    test_questions = [
        "解释一下LoRA的原理",
        "RAG的三个阶段是什么",
    ]
    test_before_after(model, tokenizer, test_questions, "训练前（基座模型）")

    # ========================================
    # Step 4: 加载并预处理数据
    # ========================================
    print(f"\n[Step 3/6] 准备训练数据")
    dataset = load_sft_data(data_path)
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    print(f"  ✓ 数据预处理完成, {len(tokenized_dataset)} 条")

    # ========================================
    # Step 5: 训练
    # ========================================
    print(f"\n[Step 4/6] 开始训练")
    print(f"  Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']} × {TRAINING_CONFIG['gradient_accumulation_steps']} = {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")

    # 自定义回调：每步打印 Loss
    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
        def on_log(self, args, state, control, logs=None, **kwargs):
            elapsed = time.time() - self.t0
            loss = logs.get("loss", "?") if logs else "?"
            print(f"  [Step {state.global_step:3d}] Loss={loss} | {elapsed:.0f}s", flush=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        logging_steps=1,
        logging_strategy="steps",
        save_steps=TRAINING_CONFIG["save_steps"],
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProgressCallback()],
    )

    trainer.train()
    print("  ✓ 训练完成")

    # ========================================
    # Step 6: 保存 + 训练后测试
    # ========================================
    print(f"\n[Step 5/6] 保存 LoRA 权重")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 检查保存的文件大小
    lora_size = 0
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            lora_size += os.path.getsize(fp)
    print(f"  ✓ 保存到: {output_dir}")
    print(f"  LoRA 权重总大小: {lora_size / 1024 / 1024:.1f} MB")
    print(f"  对比基座模型 ~6GB —— 这就是 LoRA 的魔力")

    print(f"\n[Step 6/6] 训练后测试")
    test_before_after(model, tokenizer, test_questions, "训练后（微调模型）")

    # ========================================
    # 总结
    # ========================================
    print(f"\n{'='*50}")
    print("SFT 微调完成！")
    print(f"{'='*50}")
    print(f"""
面试时你可以这样说:
  "我用 LoRA 对 Qwen2.5-3B 做了 SFT 微调。
   训练数据是 {len(dataset)} 条 LLM 技术问答对，
   可训练参数只占总参数的 {trainable/total*100:.1f}%，
   最终 LoRA 权重只有 {lora_size/1024/1024:.1f} MB。
   我在训练前后做了对比测试，可以看到模型在技术问题上回答更加准确和有结构。"
""")


if __name__ == "__main__":
    main()
