# Transformer 架构详解

## 整体结构
Transformer 由编码器(Encoder)和解码器(Decoder)组成。现代大语言模型(如GPT、Qwen)通常只使用解码器部分。

## 核心组件

### 1. 自注意力机制 (Self-Attention)
- **公式**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **Q(Query)**: 当前token想查什么
- **K(Key)**: 所有token的"索引标签"
- **V(Value)**: 所有token的实际内容
- **√d_k**: 缩放因子，防止点积过大导致softmax梯度消失
- **多头注意力(MHA)**: 用多组独立的QKV，让模型从不同角度理解上下文

### 2. 前馈网络 (FFN)
- **公式**: FFN(x) = W_down · SiLU(W_gate · x ⊙ W_up · x)
- SwiGLU激活函数：比ReLU更平滑，信息保留更多
- FFN占Transformer计算量的约70%

### 3. 层归一化 (Layer Normalization)
- 对每个样本的特征维度做标准化
- 均值为0，标准差为1
- 然后乘γ加β（可学习参数）
- 作用：稳定训练，加速收敛

### 4. 残差连接 (Residual Connection)
- x = x + SubLayer(LayerNorm(x))
- 解决深层网络的梯度消失问题
- 让模型至少能保持"什么都不做"的能力

### 5. 位置编码
- Transformer本身不具备位置感知
- RoPE(旋转位置编码)：通过旋转矩阵编码相对位置
- 优势：自然支持外推（训练1024→推理4096）

## 参数量估算
以Qwen2.5-3B为例：
- hidden_size=2048, num_layers=36, num_heads=32
- 参数量约3B，显存约6GB(FP16)
