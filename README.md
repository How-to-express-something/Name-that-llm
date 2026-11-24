# Name that LLM

Kaggle 比赛项目：[Name that LLM](https://www.kaggle.com/competitions/name-that-llm)。

用于整理比赛代码、实验笔记和方案。

比赛数据、模型权重和生成的输出文件保存在本地，不提交到 Git。

## TF-IDF 基线实验

词级 TF-IDF + LinearSVC 在 Kaggle 公开榜取得 **0.273**。本地按 150 个提示词做五折验证，平衡准确率为 **26.45%**（按提示词重抽样的 95% 区间为 24.93%–27.93%）。18 类均匀猜测的期望是 5.56%。

| 特征 | 本地平衡准确率 |
| --- | ---: |
| 仅英语停用词表中的常用词 | 19.89% |
| 单词 | 27.08% |
| 单词与相邻词对（已提交版本） | 26.45% |
| 字符 3–5 gram | 30.45% |

2,699 段折外预测中，词级主模型认对了 714 段的具体模型、1,622 段的厂商。真实来源为 `gemini-3.8-flash` 的 150 段中，57 段被判为 `gemini-3.7-flash`。这些结果支持常用词和短语携带可跨题目的来源线索，同时显示出相邻版本难以区分。

- [详细实验分析报告（下载 HTML 后用浏览器打开）](tfidf-baseline/guide.html) 包含特征对照、置信区间、混淆矩阵、特征权重、互信息分析和复现步骤。
- [Kaggle 提交 Notebook](tfidf-baseline/tfidf-baseline.ipynb) 使用词级 TF-IDF。字符特征的成绩来自本地对照实验。
- [训练与预测脚本](tfidf-baseline/baseline.py) · [按提示词验证的脚本](tfidf-baseline/evaluation.py) · [依赖版本](tfidf-baseline/requirements.txt)
