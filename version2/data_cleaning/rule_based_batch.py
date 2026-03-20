"""
基于规则的灰度数据处理脚本
根据你提供的判定规则，对灰度职位进行智能分类
"""

import pandas as pd
import numpy as np
import os
import re

# 路径配置
GRAY_ZONE_PATH = '/Users/oscar/Desktop/STAT1016GP/version2/data/gray_zone_data.csv'
OUTPUT_DIR = '/Users/oscar/Desktop/STAT1016GP/version2/data'

print("="*60)
print("基于规则的灰度数据处理")
print("="*60)

# 加载灰度数据
df = pd.read_csv(GRAY_ZONE_PATH)
print(f"\n灰度数据总量: {len(df)}")

# ============================================================
# 规则定义
# ============================================================

# AI 核心技能关键词（出现则判定为 1）
AI_SKILL_KEYWORDS = [
    'machine learning', 'deep learning', 'ml ', 'ai ', 'nlp', 'computer vision',
    'neural network', 'tensorflow', 'pytorch', 'keras', 'llm', 'large language',
    'generative', 'transformer', 'reinforcement learning', 'gan', 'diffusion',
    'autonomous', 'robotics', 'mlops', 'kubeflow', 'airflow',  # MLOps 相关
    'sagemaker', 'vertex ai', 'bedrock'  # AI 平台
]

# 非 AI 核心技能关键词（出现则判定为 0）
NON_AI_KEYWORDS = [
    'business intelligence', ' bi ', 'tableau', 'power bi', 'looker',
    'etl', 'data warehouse', 'dbt', 'snowflake', 'bigquery',
    'sql only', 'excel', 'reporting', 'dashboard'
]

# 硬编码规则（职位名称 -> 判定）
TITLE_RULES = {
    # 明确是 AI 的职位 -> 1
    'AI Software Engineer': 1,
    'AI Specialist': 1,
    'AI Consultant': 0,  # 通常是咨询类，非核心开发
    'AI Architect': 1,    # AI 架构师属于 AI 核心
    'AI Product Manager': 0,  # 产品经理，非技术开发
    'AI Lead': 1,
    'AI Manager': 0,
    'AI Director': 0,

    # Data Scientist 系列需要看技能
    'Data Scientist': None,  # 需要看技能
    'Senior Data Scientist': None,
    'Staff Data Scientist': None,
    'Principal Data Scientist': None,
    'Lead Data Scientist': None,
    'Junior Data Scientist': None,

    'AI Researcher': 1,
    'AI Analyst': 0,  # 分析类，非开发
    'AI Scientist': 1,
}


def classify_job(job_title, required_skills):
    """
    根据规则判定职位

    判定规则：
    1. 职位名称在硬编码规则中 -> 直接使用
    2. 否则根据技能关键词判断
    """
    # 清理技能字符串
    if not isinstance(required_skills, str):
        skills_lower = ""
    else:
        skills_lower = required_skills.lower()

    # 1. 检查硬编码规则
    if job_title in TITLE_RULES:
        rule = TITLE_RULES[job_title]
        if rule is not None:
            return rule, f"规则匹配: {job_title}"

    # 2. 检查 AI 核心技能关键词
    ai_match = any(kw in skills_lower for kw in AI_SKILL_KEYWORDS)
    non_ai_match = any(kw in skills_lower for kw in NON_AI_KEYWORDS)

    # Data Scientist 系列特殊处理
    if 'data scientist' in job_title.lower():
        if ai_match:
            return 1, "Data Scientist with ML skills"
        elif non_ai_match and not ai_match:
            return 0, "Data Scientist BI/reporting only"
        else:
            # 默认偏向 1（Data Scientist 通常需要 ML）
            return 1, "Data Scientist default"

    # 3. 其他情况根据关键词判断
    if ai_match and not non_ai_match:
        return 1, "AI skills found"
    elif non_ai_match and not ai_match:
        return 0, "Non-AI skills only"
    elif ai_match and non_ai_match:
        # 两者都有，看 AI 是否更核心
        return 1, "Mixed but AI present"
    else:
        # 没有任何匹配，默认 0
        return 0, "No clear AI indicator"


# ============================================================
# 处理灰度数据
# ============================================================
print("\n" + "="*60)
print("开始规则分类...")
print("="*60)

results = []
for i, (_, row) in enumerate(df.iterrows()):
    job_title = row['job_title']
    required_skills = row['required_skills']

    ai_tag, reason = classify_job(job_title, required_skills)

    results.append({
        'job_id': row['job_id'],
        'job_title': job_title,
        'required_skills': required_skills,
        'rule_ai_tag': ai_tag,
        'rule_reason': reason
    })

    if (i + 1) % 500 == 0:
        print(f"  已处理: {i+1}/{len(df)}")

print(f"  已处理: {len(df)}/{len(df)}")

# 统计结果
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("分类结果统计")
print("="*60)
print("\n按判定结果分布:")
print(results_df['rule_ai_tag'].value_counts())

print("\n按判定理由分布:")
print(results_df['rule_reason'].value_counts().head(10))

# 保存结果
results_path = os.path.join(OUTPUT_DIR, 'gray_zone_ruled.csv')
results_df.to_csv(results_path, index=False)
print(f"\n规则分类结果已保存: {results_path}")

# ============================================================
# 合并回主数据
# ============================================================
print("\n" + "="*60)
print("合并结果到主数据...")
print("="*60)

# 读取原始清洗后的数据
main_df = pd.read_csv('/Users/oscar/Desktop/STAT1016GP/version2/data/cleaned_ai_job_data_v2.csv')

# 创建 job_id 到 ai_tag 的映射
job_to_tag = dict(zip(results_df['job_id'], results_df['rule_ai_tag']))

# 更新主数据中 gray zone 部分的判定
for idx, row in df.iterrows():
    job_id = row['job_id']
    if job_id in job_to_tag:
        # 找到主数据中对应的行（通过原始索引）
        main_idx = main_df.index[idx] if idx < len(main_df) else None
        if main_idx is not None:
            main_df.loc[main_idx, 'AI_Skill_Requirement'] = job_to_tag[job_id]

# 保存更新后的数据
main_df.to_csv('/Users/oscar/Desktop/STAT1016GP/version2/data/cleaned_ai_job_data_v2.csv', index=False)

print(f"\n最终数据统计:")
print(main_df['AI_Skill_Requirement'].value_counts())
print(f"\nAI Skill Required: {main_df['AI_Skill_Requirement'].sum()} ({main_df['AI_Skill_Requirement'].mean()*100:.1f}%)")
print(f"Non-AI: {(1-main_df['AI_Skill_Requirement']).sum()} ({(1-main_df['AI_Skill_Requirement'].mean())*100:.1f}%)")

# ============================================================
# 重新运行分析
# ============================================================
print("\n" + "="*60)
print("重新运行统计分析...")
print("="*60)

# T-Test
salary_ai = main_df[main_df['AI_Skill_Requirement'] == 1]['Log_Salary']
salary_no_ai = main_df[main_df['AI_Skill_Requirement'] == 0]['Log_Salary']

from scipy import stats

t_stat, p_value = stats.ttest_ind(salary_ai, salary_no_ai)

print(f"\nT-Test 结果:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"\n  AI 岗位平均薪资: ${np.exp(salary_ai.mean()):,.2f}")
print(f"  非AI岗位平均薪资: ${np.exp(salary_no_ai.mean()):,.2f}")

# 计算溢价
premium = (np.exp(salary_ai.mean() - salary_no_ai.mean()) - 1) * 100
print(f"  AI 技能溢价: {premium:.2f}%")

if p_value < 0.05:
    print(f"\n结论: AI 技能对薪资有显著影响 (p < 0.05)")
else:
    print(f"\n结论: AI 技能对薪资无显著影响 (p >= 0.05)")

print("\n" + "="*60)
print("处理完成!")
print("="*60)
