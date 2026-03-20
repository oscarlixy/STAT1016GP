# AI Job Market Salary Analysis - Version 2

## 项目概述

Version 2 在 Version 1 基础上对 **AI 技能岗位判定方法** 进行了重大升级，采用三层分类体系：

```
┌─────────────────────────────────────────────────────────────────┐
│                    三层 AI 技能分类体系                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: 正则粗筛                                                │
│  ├── AI 核心关键词: AI, ML, Deep Learning, NLP, Computer Vision...  │
│  ├── 非 AI 关键词: Software Engineer, Backend, Data Analyst...     │
│  └── 灰度关键词: AI Specialist, AI Architect, Data Scientist...   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: 灰度数据提取                                            │
│  └── 将无法确定的岗位 (tag=None) 提取为独立 DataFrame              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: API 批处理                                             │
│  └── 调用 Gemini API，使用 LLM 进行精确判定                        │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
STAT1016GP/
├── version1/                          # Version 1 (原始版本)
│   ├── data/
│   ├── visualization/
│   ├── mlr/
│   └── t_test_analysis/
├── version2/                          # Version 2 (升级版本)
│   ├── data_cleaning/
│   │   └── data_cleaning.py           # 三层分类核心脚本
│   ├── data/
│   │   ├── ai_job_dataset.csv         # 原始数据
│   │   ├── gray_zone_data.csv         # 灰度数据 (Layer 2 输出)
│   │   └── cleaned_ai_job_data_v2.csv # 清洗后数据 (最终输出)
│   ├── visualization/
│   │   └── eda_visualization.py
│   ├── mlr/
│   │   └── mlr_regression.py
│   ├── t_test_analysis/
│   │   └── ttest_analysis.py
│   ├── config_template.py             # 配置模板
│   └── README.md
└── data/                              # V1 数据 (保持兼容)
    ├── ai_job_dataset.csv
    └── cleaned_ai_job_data.csv
```

## 核心改进

### 1. Layer 1: 正则粗筛

**AI 核心关键词 (ai_tag = 1):**
- AI, ML, MLOps, Machine Learning, Deep Learning
- NLP, Computer Vision, Neural Network
- AI Research, ML Infrastructure, ML Platform
- Autonomous, Robotics, Generative AI, LLM
- Reinforcement Learning, Transformer

**非 AI 关键词 (ai_tag = 0):**
- Software Engineer, Backend, Frontend, Full Stack
- Data Analyst, Business Analyst, BI Engineer
- DevOps, SRE, Cloud Engineer
- Product Manager, Project Manager

**灰度关键词 (需要 LLM 判定):**
- AI Specialist, AI Architect, AI Consultant
- Data Scientist (所有级别)
- Principal AI Lead, Senior AI Manager

### 2. Layer 2: 灰度数据提取

正则无法确定的岗位被提取到 `gray_zone_data.csv`，包含：
- `_gray_zone_reason`: 灰度原因 (如 "Gray zone - needs LLM")
- 原始数据的所有字段

### 3. Layer 3: API 批处理

调用 Gemini API 进行精确判定：

**提示词模板:**
```
你是一位资深的人工智能与人力资源数据科学家...
判定规则:
- 输出 1: 核心涉及 AI 底层研发、模型训练、AI 基建
- 输出 0: 传统软件工程、常规数据分析、普通运维

输出 JSON: {"ai_tag": <1或0>, "reason": "<20字判定理由>"}
```

## 使用方法

### 1. 仅运行正则分类 (不调用 API)

```bash
cd version2
python data_cleaning/data_cleaning.py
```

### 2. 调用 API 处理灰度数据

```bash
# 设置 API Key
export GEMINI_API_KEY="your_api_key"

# 运行完整流程
python data_cleaning/data_cleaning.py \
    --api-key "$GEMINI_API_KEY" \
    --use-api \
    --rate-limit 5.0
```

### 3. 单独处理已存在的灰度数据

```bash
python data_cleaning/data_cleaning.py \
    --api-key "$GEMINI_API_KEY" \
    --process-existing
```

## 运行分析

```bash
# 数据清洗
python data_cleaning/data_cleaning.py --api-key "KEY" --use-api

# 可视化
python visualization/eda_visualization.py

# 多元回归
python mlr/mlr_regression.py

# T 检验
python t_test_analysis/ttest_analysis.py
```

## AI 岗位判定规则详解

### 判定为 1 (需要核心 AI 技能)

| 情况 | 示例 |
|------|------|
| 岗位名称包含 AI/ML 核心词 | AI Research Scientist, ML Engineer, NLP Engineer |
| 涉及模型训练/优化 | Machine Learning Engineer, Deep Learning Specialist |
| AI 基础设施/平台 | ML Infrastructure Engineer, MLOps Engineer |
| 特殊例外 (底层语言无高层框架) | NLP Engineer (Scala/C++/Python), CV Researcher (C++/CUDA) |

### 判定为 0 (不需要核心 AI 技能)

| 情况 | 示例 |
|------|------|
| 传统软件工程 | Software Engineer, Backend Developer, Web Developer |
| 常规数据分析 | Data Analyst, BI Analyst, Business Analyst |
| 普通运维开发 | DevOps Engineer, SRE, Cloud Engineer |
| 技能栈通用但非 AI 核心 | Python Developer, Docker/K8s 专家 (无 AI 模型相关) |

## 技术栈

- Python 3.x
- pandas, numpy
- statsmodels, scipy
- matplotlib, seaborn
- Gemini API (可选)

## 安装依赖

```bash
pip install pandas numpy statsmodels scipy matplotlib seaborn
```

## 注意事项

1. **API 配额**: Gemini API 有免费配额限制，注意设置 `--rate-limit`
2. **灰度比例**: 通常约 15-25% 的岗位需要 LLM 判定
3. **默认处理**: API 失败的岗位默认标记为 0 (保守策略)
4. **API Key**: 从 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取

## 作者

- Oscar

## 许可证

MIT License
