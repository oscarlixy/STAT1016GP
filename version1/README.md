# AI Job Market Salary Analysis

全球 AI 就业市场趋势与薪资分析项目

## 📋 项目概述

本项目使用多元线性回归 (MLR) 分析 AI 技能、工作经验年限和行业对薪资的影响。

**目标回归模型**:
$$\log(Salary) = \beta_{0} + \beta_{1}(AI\_Skill) + \beta_{2}(Experience) + \beta_{3}(Industry) + \epsilon$$

## 📁 项目结构

```
STAT1016GP/
├── data/
│   ├── ai_job_dataset.csv          # 原始数据集
│   └── cleaned_ai_job_data.csv    # 清洗后的数据
├── visualization/
│   ├── eda_visualization.py       # EDA 可视化脚本
│   ├── correlation_heatmap.png    # 相关性热力图
│   └── boxplot_salary_comparison.png  # 薪资分布箱线图
├── mlr/
│   ├── mlr_regression.py          # 多元线性回归脚本
│   └── regression_results.txt     # 回归结果
├── t_test_analysis/
│   ├── ttest_analysis.py          # T检验分析
│   ├── ttest_results.txt
│   └── ttest_summary.csv
└── README.md
```

## 🔬 分析方法

### 1. 数据清洗与特征工程
- 薪资统一转换为 USD
- 对数变换 (Log_Salary)
- 经验值标准化 (Experience_Scaled)
- 行业虚拟变量编码

### 2. 探索性数据分析 (EDA)
- **相关性热力图**: 展示 Log_Salary、Experience_Scaled 和 AI_Skill_Requirement 之间的相关性
- **箱线图对比**: 比较需要/不需要 AI 技能的薪资分布

### 3. 多元线性回归 (MLR)
使用 statsmodels 库进行 OLS 回归分析

## 📊 核心结果

| 指标 | 数值 |
|------|------|
| **R-squared** | 56.38% |
| **AI技能溢价** | -0.44% (不显著, p=0.430) |
| **经验系数** | 0.3601 (显著, p<0.001) |

### 主要发现
- **工作经验**是薪资的最主要决定因素 (r=0.75)
- **AI技能**在控制经验和行业后，**无显著薪资溢价** (p=0.430)
- **行业**对薪资无显著影响

## 🛠️ 技术栈

- Python 3.x
- pandas, numpy
- statsmodels
- scipy
- matplotlib, seaborn

## 📦 运行方式

```bash
# 安装依赖
pip install pandas numpy statsmodels scipy matplotlib seaborn

# 运行 EDA 可视化
python visualization/eda_visualization.py

# 运行多元线性回归
python mlr/mlr_regression.py

# 运行 T 检验分析
python t_test_analysis/ttest_analysis.py
```

## 📝 作者

- Oscar

## 📄 许可证

MIT License
