# NLP Project: StreamingLLM 优化测试报告

本项目致力于在自然语言处理(NLP)任务中，针对大语言模型在处理超长文本流极易出现的**内存溢出 (OOM)**和**注意力崩溃**问题，复现并测试了 `StreamingLLM` 的优化效果。

## 1. 测试基础信息
* **测试模型**: `EleutherAI/pythia-70m-deduped` (Pythia-70m)
* **评估数据集**: `PG-19` (长篇小说文本) 和 `Wikitext-2` (维基百科文章)
* **核心脚本**: `test.py`

## 2. 环境安装

请使用 `conda` 创建隔离的虚拟环境，并安装对应依赖：

```bash
# 1. 创建并激活 conda 虚拟环境
conda create -yn streaming python=3.8
conda activate streaming

# 2. 安装 PyTorch 及其相关生态库 (请根据您的硬件架构选择对应的 cuda/mps 选项)
pip install torch torchvision torchaudio

# 3. 安装其他必需依赖
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

# 4. 以开发模式安装项目依赖包 (使得 streaming_llm 可以被引用)
python setup.py develop
```

## 3. 运行代码

环境配置完成后，直接使用 Python 运行评估脚本即可：

```bash
python test.py
```

## 4. 测试与优化总结报告

在处理长度达到了 4000 tokens 的连续流数据测试中，我们对比了原版无缓存控制的模型 (Baseline) 以及加入优化策略后的模型 (StreamingLLM)。核心指标反馈如下：

### 4.1 延迟与吞吐量 (速度指标)

| 指标 | Baseline (基线) | StreamingLLM (优化后) | 变化 |
| :--- | :--- | :--- | :--- |
| **首字输出延迟 (TTFT)** | `545.16 ms` | `31.98 ms` | 🚀 **提升 17倍** |
| **单字生成耗时 (TPOT)** | `38.23 ms/token` | `59.14 ms/token` | 略微增加 |
| **整体吞吐量** | `21.67 tokens/s` | `17.03 tokens/s` | 小幅下降 |

**分析**: StreamingLLM 极大地降低了模型处理并返回首字的时间 (TTFT)，使得其极具响应优势。后续字元的生成速度 (TPOT) 有少量损耗，这是因为 StreamingLLM 在每一步生成都在高强度地执行 KV Cache 的截断拼接逻辑（保留 Attention Sinks 丢弃中间无用历史）。

### 4.2 语言建模困惑度 (PPL 指标)


| 数据集 | Baseline PPL | StreamingLLM PPL | 优化效果 |
| :--- | :--- | :--- | :--- |
| **PG-19** | `68.03` | `40.36` | 📉 **显著下降，改善明显** |
| **Wikitext-2** | `84.55` | `51.67` | 📉 **显著下降，改善明显** |

**分析**:
在超过模型原生训练窗口的长流文本中，Baseline 很快因为前序上下文缓存爆满，注意力完全崩溃从而导致 PPL 急剧上升。而 **StreamingLLM 结合对 Attention Sinks（注意力汇聚点/前几个主要Token）的永久固定**，强力锁定了注意力的锚定焦点，在只占据恒定极小内存的基础上，完美找回了模型的长上下文语言建模能力，大幅降低了崩溃造成的 PPL 数值。这证明了其长生命周期对话调度的可靠能力。