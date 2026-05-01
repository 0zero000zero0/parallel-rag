# Adaptive Parallel O1

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Parallel-RAG 是一个创新的 **Agentic RAG** 框架，旨在解决传统 Search-O1 类模型在引入外部知识后，因推理分布（CoT）被检索结果干扰而导致的 **查询重复** 与 **推理不确定性** 问题。


## ✨ 核心特性

-   **🧭 自适应并行检索**：导航器（Navigator Agent）每轮动态生成多个抽象的检索方向（Search Directions），由路径代理（Path Agent）并行转化为具体查询，一次性覆盖多种信息需求。
-   **🔎 消除查询冗余**：通过全局规划和并行执行，有效避免了因检索知识分布偏移而导致的重复或无效查询。
-   **🧠 内外知识协同**：引入“全局精炼”（Global Refine Agent）阶段，对并行检索到的文档进行联合分析、去重与提炼，生成结构化的信息摘要，平滑地融入推理链。
-   **⚡ 高性能与低延迟**：并行执行路径代理和检索请求，相比串行的 Search-O1，**运行时间减少 20%**，同时显著提升答案质量。
-   **🔄 灵活可配置**：支持为不同代理（导航器、路径代理、全局精炼器）独立配置模型、参数及提示模板，适应各类 LLM 环境。


## 🚀 快速开始

### 环境准备

```bash
git clone https://github.com/0zero000zero0/parallel-rag.git
cd parallel-rag
pip install -r requirements.txt
```

### 配置检索服务

本项目依赖独立[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)的检索服务。首先需要下载[wikipedia 2018 en corpus和index](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus)，检索器[e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)。然后在`retriever_config.yaml`中配置检索器和语料库路径：

```yaml
retrieval_method: "dense"
retrieval_model_path: "/path/to/e5-base-v2"
index_path: "/path/to/e5_flat_inner.index"
corpus_path: "/path/to/wiki18_100w.jsonl"
retrieval_topk: 5
retrieval_batch_size: 256
faiss_gpu: True
```
启动retrieval server:
```bash
./bash/retrieval-server.sh
```

### 运行实验
先启动分别启动vllm server和retrieval server:

```bash
./bash/retrieval-server.sh
./bash/vllm-server.sh
```

运行实验：
```bash
./bash/run_adaptive_parallel_o1.sh
```


### 主要脚本说明

| 脚本                           | 功能描述                                                       |
| :----------------------------- | :------------------------------------------------------------- |
| `run_adaptive_parallel_o1.py`  | **核心实现**：自适应并行 O1 框架的主入口。                     |
| `run_parallel_rag.py`          | 固定并行检索版本的 RAG 实现。                                  |
| `run_search_o1.py`             | 标准 Search-O1 基线（用于对比）。                               |
| `run_cot.py` .... | 其他基线方法（Chain-of-Thought， 基础 RAG）。                  |
| `evaluate.py`                  | 在指定数据集上运行评估，计算准确率等指标。                     |
| `gather_metric.py`             | 收集并汇总多次实验的指标，生成对比表格。                       |
| `retrieval-server/`            | 独立的检索服务端代码，支持文档批量搜索。                       |


## 📄 项目结构

```
parallel-rag/
├── src/                        # 核心源码
│   ├── clients.py              # LLM、检索器等客户端封装
│   ├── prompted_generation_base.py # 基础生成类
│   └── adaptive_parallel_o1.py # Adaptive Parallel O1 核心实现
├── retrieval-server/           # 独立检索服务
├── config/                     # 配置文件目录
├── bash/                       # 启动脚本
├── evaluate.py                 # 评估
├── gather_metric.py            # 指标汇总工具
├── run_*.py                    # 各种运行入口（基线、主要方法）
└── README.md                   # 本文件
```


## 🤝 贡献

欢迎通过 Issue 或 Pull Request 提出改进建议、报告问题或贡献代码。

## 📧 联系

项目维护者：Dingwen Zhang (0zero000zero0)
