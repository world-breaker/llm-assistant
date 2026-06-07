# 🤖 LLM 技术面试智能助手

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--3B-orange)](https://huggingface.co/Qwen)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> 基于 **Qwen2.5-3B + RAG + LoRA 微调** 的 LLM 技术问答系统  
> 面试项目展示 | 华科软工硕士毕设配套

## 📸 效果演示

![demo](docs/demo.png)

## 🧠 项目概述

面向 LLM 技术面试的智能问答助手。用户提问 AI/大模型相关问题（Transformer、LoRA、RAG、RLHF、部署优化等），系统从技术知识库中检索相关文档，结合微调后的 Qwen2.5-3B 模型生成准确、有条理的回答。

## 🏗️ 技术架构

```
用户提问 → 向量检索(语义+关键词) → ChromaDB → Prompt构造 → Qwen2.5-3B + LoRA → 回答
```

## ✨ 核心特性

- **RAG 检索增强**：混合检索（语义相似度 + 关键词匹配），减少模型幻觉
- **LoRA 微调**：18条技术问答对微调，可训练参数仅 0.5%，权重仅 68MB
- **Gradio 网页**：一键启动，支持多轮对话，展示检索到的参考文档
- **全本地运行**：无需 API Key，Qwen 3B 在 RTX 5060 (8GB) 上流畅运行

## 📁 项目结构

```
llm_interview_assistant/
├── data/
│   ├── knowledge/              # 6篇LLM技术知识文档
│   └── sft_train.json          # 18条SFT训练数据
├── src/
│   ├── build_kb.py             # 构建知识库：分块→向量化→ChromaDB
│   ├── rag_pipeline.py         # RAG核心管线：检索→Prompt→生成
│   ├── sft_train.py            # LoRA微调脚本
│   └── app.py                  # Gradio网页Demo
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 环境要求
- Python 3.12
- NVIDIA GPU (8GB+ VRAM)
- CUDA 12.8

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 构建知识库
```bash
cd src
python build_kb.py
```

### 3. LoRA 微调（可选）
```bash
python sft_train.py
```

> 已提供预训练数据 `data/sft_train.json`，5个epoch约需5-10分钟。

### 4. 启动服务
```bash
python app.py
```

浏览器自动打开 `http://127.0.0.1:7860`

## 🎯 面试讲解要点

### 这个项目展示了什么能力？

| 能力维度 | 具体体现 |
|---|---|
| **RAG 全链路** | 文档处理 → 向量化 → 混合检索 → Prompt工程 → 生成 |
| **模型微调** | LoRA 参数高效微调，0.5%可训练参数，68MB权重 |
| **工程化能力** | 模块化设计、配置管理、错误处理、GPU/CPU自适应 |
| **产品思维** | 命令行工具 → Gradio网页，可演示可交互 |

### 面试官常问的10个问题

1. **为什么用 Qwen 做 Embedding？** → 复用已有模型，中文语义好。但实测区分度不够，生产环境推荐 BGE
2. **分块大小为什么是600？** → 600字≈一个完整技术概念，太大检索不准，太小碎片化
3. **LoRA 的 r=8 怎么选的？** → r=4~16都是合理范围，8是性价比最优
4. **18条数据够微调吗？** → SFT的核心是教会格式和风格，不是灌输知识。知识靠预训练+RAG
5. **ChromaDB vs FAISS？** → Chroma自带文档存储+持久化，demo开箱即用
6. **为什么用贪心解码不用采样？** → 技术问答追求准确性和确定性，贪心更合适且速度更快
7. **怎么评估 RAG 效果？** → 检索命中率 + 回答准确性 + 幻觉率
8. **模型在 GPU 还是 CPU？** → RTX 5060 Laptop (8GB)，FP16推理，150 tokens ≈ 5-10秒
9. **如果知识库更新了怎么办？** → 重新运行 `build_kb.py` 即可，增量更新
10. **怎么处理知识库中没有的问题？** → 设置相似度阈值0.35，低于阈值跳过检索，让模型基于自身知识回答

## 📊 模型性能

| 指标 | 数值 |
|---|---|
| 基座模型 | Qwen2.5-3B-Instruct |
| 模型大小 | ~6GB (FP16) |
| LoRA 可训练参数 | 0.5% |
| LoRA 权重大小 | 68.1 MB |
| 单次推理耗时 | 3-10秒 |
| GPU 显存占用 | ~6.5GB |

## 🔧 技术栈

- **模型**: Qwen2.5-3B-Instruct (ModelScope)
- **嵌入**: Qwen 隐藏状态 (2048维)
- **向量库**: ChromaDB
- **微调**: LoRA (PEFT) + Transformers Trainer
- **界面**: Gradio ChatInterface
- **硬件**: NVIDIA RTX 5060 Laptop (8GB VRAM)

## 📝 License

MIT
