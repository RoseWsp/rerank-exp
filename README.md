# rerank-exp

一次重排序对照实验：**专用 Cross-Encoder Reranker（Qwen3-Reranker）vs TypeSafe jev（System One 决策模型 + Noul）**。

两者都在**完全相同**的 BM25 候选短名单上打分，只有"用谁重排"这一个变量不同。

数据集用 CLERC（美国联邦法院检索）的一小片，与 [TypeSafe 官方 Re-ranking cookbook](https://docs.typesafe.ai) 的设置一致。

---

## 结果速览

40 个查询 × 30 篇候选 = 1200 个 (query, candidate) 对。BM25 短名单对全部选手完全相同。

| 方法 | Top-1 | Top-5 | Top-10 | AUC | 平均 gold 排名 | 秒/query | 成本 |
|---|---|---|---|---|---|---|---|
| BM25（不重排，基线） | 5% | 15% | 37.5% | 0.528 | 14.68 | — | — |
| Qwen3-Reranker-0.6B | 2% | 12.5% | 37.5% | 0.556 | 13.93 | 3.23 | ~$0 |
| Qwen3-Reranker-4B | 7.5% | 27.5% | 47.5% | 0.678 | 11.05 | 10.60 | ~$0 |
| Qwen3-Reranker-8B | 7.5% | 20% | 47.5% | 0.634 | 11.70 | 111.70 | ~$0 |
| Qwen3-Reranker-8B (8bit) | 5% | 20% | 47.5% | 0.637 | 11.40 | 15.31 | ~$0 |
| **TypeSafe jev-latest** | **32.5%** | **50%** | **67.5%** | **0.790** | **7.45** | 0.83 | $0.0704 |

- **AUC**：随机取一个正确项、一个错误项，正确项分数更高的概率（0.5 = 瞎猜）。
- **秒/query**：Qwen 是本地单卡（RTX 5060 Ti 16GB），jev 是云端 API + 12 路并发，**两者不具备严格可比性**。
- jev 用的是 SDK 默认的 `jev-latest`（= `jev-1.13.0`）；官方 cookbook 用的是 `jev-1.12`，报的 Top-1 是 18%。
- 完整指标见 [`artifacts/results.json`](artifacts/results.json)。

图表：[`artifacts/chart_accuracy.png`](artifacts/chart_accuracy.png)（Top-1/5/10 对比）、[`artifacts/chart_latency.png`](artifacts/chart_latency.png)（单 query 耗时）。

---

## 关于两种"重排序"的区别（简述）

- **专用 Reranker（Qwen3-Reranker）**：一个为"判相关性"微调的判别器。把 `(指令, 查询, 文档)` 拼成模板，取最后一个 token 的 logits，在 `"yes"`/`"no"` 上做 softmax 得到分数。
- **TypeSafe jev**：一个 **System One 决策模型**（**不生成文本、不做长推理**），用 **RLCD**（面向校准决策的强化学习）训练，返回**校准概率**。你写一道 yes/no 题（**Noul**）+ true/false 的 criteria，它直接返回 0~1 的概率当分数。

两者**同属"专用打分器"**，差别在**输出契约**（校准概率 vs 判别分数）和**任务定义方式**（自然语言 criteria vs 训练时的相关性模式）。

---

## 目录结构

```
rerank-exp/
├── common.py           # CLERC 切片 + BM25 候选 + 指标（共享）
├── run_qwen.py         # Qwen3-Reranker 打分
├── run_typesafe.py     # TypeSafe jev + Noul 打分
├── eval.py             # 汇总所有 *_scores*.json，出表 + 出图
├── requirements.txt
├── models/             # 本地模型权重（.gitignore，需自行下载）
└── artifacts/          # 产出：切片、候选、得分、结果、图表
```

---

## 环境准备

需要 Python 3.11+。建议用虚拟环境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> Blackwell 显卡（如 RTX 5060 Ti，sm_120）需要 `torch` 的 cu128 及以上版本，`requirements.txt` 已按此说明。

---

## 下载模型（可选，仅跑 Qwen 时需要）

权重有几十 GB，**不在仓库里**，请自行下载到 `models/`：

```bash
# 从 ModelScope 克隆（国内较快）
git clone https://www.modelscope.cn/Qwen/Qwen3-Reranker-4B.git models/Qwen3-Reranker-4B
git clone https://www.modelscope.cn/Qwen/Qwen3-Reranker-8B.git models/Qwen3-Reranker-8B

# 或从 HuggingFace
git clone https://huggingface.co/Qwen/Qwen3-Reranker-0.6B models/Qwen3-Reranker-0.6B
```

`run_qwen.py` 会自动探测 `models/<模型名>/` 目录；没有则回退到 HuggingFace id 在线加载。

---

## 运行

**1. 数据集与 BM25 候选**（首次运行 `common.py` 相关步骤会自动下载 CLERC 并缓存到 `artifacts/`，无需单独命令）。

**2. Qwen3-Reranker**：

```bash
python run_qwen.py --size 0.6B
python run_qwen.py --size 4B
# 8B 显存不够时：量化 + CPU 卸载
python run_qwen.py --size 8B --quant 8bit --batch 4
python run_qwen.py --size 8B --device-map auto --batch 4
```

常用参数：`--size {0.6B,4B,8B}`、`--quant {none,8bit,4bit}`、`--device-map {cuda,auto}`、`--batch`、`--max-len`、`--label`（输出文件后缀）、`--model-path`（指定本地目录）。

输出：`artifacts/qwen_scores_<label>.json`。

**3. TypeSafe jev**：

在 `rerank-exp/` 下放一个 `.env`（**不要提交**）：

```
TYPESAFE_API_KEY=你的key
```

然后：

```bash
python run_typesafe.py
```

结果缓存到 `artifacts/typesafe_cache.json`，重跑不花钱。输出：`artifacts/typesafe_scores.json`。

> 可用环境变量覆盖：`TYPESAFE_MODEL`（默认 `jev-latest`）、`TYPESAFE_ENDPOINT`。

**4. 评测 + 出图**：

```bash
python eval.py
```

自动发现 `artifacts/` 下所有 `*_scores*.json`，加 BM25 基线一起比较，写出 `artifacts/results.json`、`chart_accuracy.png`、`chart_latency.png`。**先删掉不想参与比较的得分文件即可。**

---

## 复现说明

- **随机性可控**：切片用固定 `SEED=0`，语料按 id 排序，保证每次迭代顺序一致。BM25 基线（Top-1 5% / Top-5 15% / Top-10 38%）与官方 cookbook 完全吻合。
- **候选集固定**：所有选手重排的都是同一批 BM25 Top-30，保证可比。
- **jev 的题目是手写的**，`criteria` 写得好坏会显著影响结果。
- **样本量小**（40 条查询、单一数据集），请重点看**方法之间的相对差距**，不要过度外推绝对数值。

---

## License

实验代码仅供研究参考。
