"""
精简版 API 批处理脚本
只处理 Gray Zone 数据中的关键争议案例
"""

import pandas as pd
import json
import urllib.request
import urllib.error
import time
import os
from collections import Counter

# 配置
API_KEY = "AIzaSyBNWGjGBgbgpIwjCJ1p7rIdLS09H_maoFY"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GRAY_ZONE_PATH = '/Users/oscar/Desktop/STAT1016GP/version2/data/gray_zone_data.csv'
OUTPUT_PATH = '/Users/oscar/Desktop/STAT1016GP/version2/data/gray_zone_processed.csv'
RATE_LIMIT = 5.0  # 每条记录间隔秒数
BATCH_SIZE = 500  # 每次处理的批次数
MAX_RETRIES = 5  # 最大重试次数
INITIAL_DELAY = 5.0  # 初始延迟秒数

PROMPT_TEMPLATE = '''你是一位资深的人工智能与人力资源数据科学家。你的任务是判定该岗位是否需要"核心AI技能"。

判定规则：
输出1：岗位核心涉及AI底层研发、模型训练、AI基建(MLOps)或AI核心应用开发
输出0：传统软件工程、常规数据分析(BI)、普通运维开发

特例注意：
- NLP Engineer, ML Infrastructure Engineer 等高级AI岗位，即使只列底层语言(C++, Scala, Python)，也判定为1
- Data Scientist：根据技能栈判断，涉及ML/DL/NLP则判定1，仅SQL/BI/可视化则判定0

职位名称: {job_title}
技能要求: {required_skills}

输出格式（仅JSON，无任何其他文本）：
{{"ai_tag": 1或0, "reason": "不超过20字理由"}}
'''

def call_api(job_title, required_skills):
    """调用 Gemini API"""
    prompt = PROMPT_TEMPLATE.format(job_title=job_title, required_skills=required_skills)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 80}
    }
    url = f"{BASE_URL}?key={API_KEY}"

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                # 清理可能的 markdown
                if text.startswith('```'):
                    text = text.split('\n', 1)[1]
                if text.endswith('```'):
                    text = text.rsplit('\n', 1)[0]
                return json.loads(text.strip())
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limit
                delay = INITIAL_DELAY * (2 ** attempt)
                print(f"    速率限制，等待 {delay:.1f}s...")
                time.sleep(delay)
                continue
            elif attempt < MAX_RETRIES - 1:
                delay = INITIAL_DELAY * (2 ** attempt)
                print(f"    HTTP错误 {e.code}, 等待 {delay:.1f}s...")
                time.sleep(delay)
                continue
            return None
        except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_DELAY * (2 ** attempt)
                print(f"    错误: {str(e)[:30]}, 等待 {delay:.1f}s...")
                time.sleep(delay)
                continue
            return None
    return None

def main():
    print("="*60)
    print("精简版 API 批处理")
    print("="*60)

    # 加载灰度数据
    df = pd.read_csv(GRAY_ZONE_PATH)
    print(f"\n灰度数据总量: {len(df)}")

    # 分析职位分布
    titles = df['job_title'].value_counts()
    print("\n职位分布 (Top 15):")
    for title, count in titles.head(15).items():
        print(f"  {title}: {count}")

    # 关键职位列表（最需要 LLM 判定的）
    KEY_TITLES = [
        'AI Software Engineer', 'AI Specialist', 'AI Consultant', 'AI Architect',
        'AI Product Manager', 'AI Lead', 'AI Manager', 'AI Director',
        'Data Scientist', 'Senior Data Scientist', 'Staff Data Scientist',
        'Principal Data Scientist', 'Lead Data Scientist', 'Junior Data Scientist',
        'AI Researcher', 'AI Analyst', 'AI Scientist'
    ]

    # 筛选关键职位
    key_df = df[df['job_title'].isin(KEY_TITLES)].copy()
    print(f"\n关键争议职位数量: {len(key_df)}")

    # 统计
    print(f"  - AI Software Engineer: {len(key_df[key_df['job_title']=='AI Software Engineer'])}")
    print(f"  - AI Specialist: {len(key_df[key_df['job_title']=='AI Specialist'])}")
    print(f"  - AI Consultant: {len(key_df[key_df['job_title']=='AI Consultant'])}")
    print(f"  - AI Architect: {len(key_df[key_df['job_title']=='AI Architect'])}")
    print(f"  - Data Scientist (all levels): {len(key_df[key_df['job_title'].str.contains('Data Scientist', na=False)])}")
    print(f"  - AI Product Manager: {len(key_df[key_df['job_title']=='AI Product Manager'])}")

    # 处理
    print("\n" + "="*60)
    print("开始 API 处理...")
    print("="*60)

    results = []
    total = len(key_df)
    success = 0
    failed = 0

    for i, (_, row) in enumerate(key_df.iterrows()):
        title = row['job_title']
        skills = row['required_skills'] if isinstance(row['required_skills'], str) else ""

        print(f"[{i+1}/{total}] {title[:40]}...", end=' ')

        result = call_api(title, skills)

        if result:
            ai_tag = result.get('ai_tag', 0)
            reason = result.get('reason', '')
            print(f"-> {ai_tag} ({reason})")
            success += 1
        else:
            ai_tag = 0
            reason = 'API failed'
            print("-> FAILED (default 0)")
            failed += 1

        results.append({
            'job_id': row['job_id'],
            'job_title': title,
            'required_skills': skills,
            'api_ai_tag': ai_tag,
            'api_reason': reason
        })

        # 速率限制
        if i < total - 1:
            time.sleep(RATE_LIMIT)

        # 每500条显示进度
        if (i + 1) % 100 == 0:
            print(f"\n  进度: {i+1}/{total} (成功: {success}, 失败: {failed})")

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    # 统计结果
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
    print(f"总处理: {total}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"\nAPI 判定分布:")
    print(results_df['api_ai_tag'].value_counts())
    print(f"\n结果已保存: {OUTPUT_PATH}")

    # 合并回主数据
    print("\n" + "="*60)
    print("合并结果到主数据...")
    print("="*60)

    # 读取原始清洗后的数据
    main_df = pd.read_csv('/Users/oscar/Desktop/STAT1016GP/version2/data/cleaned_ai_job_data_v2.csv')

    # 读取带 gray zone 标注的原始数据
    full_gray = df.copy()

    # 用 API 结果更新 gray zone
    for _, result_row in results_df.iterrows():
        mask = full_gray['job_id'] == result_row['job_id']
        if mask.any():
            full_gray.loc[mask, '_layer1_tag'] = result_row['api_ai_tag']
            full_gray.loc[mask, '_layer1_reason'] = result_row['api_reason']
            full_gray.loc[mask, '_layer1_confidence'] = 'api'

    # 重新计算 AI_Skill_Requirement
    full_gray['AI_Skill_Requirement'] = full_gray['_layer1_tag'].fillna(0).astype(int)

    # 重新生成 Log_Salary 和 Experience_Scaled
    import numpy as np
    full_gray['Log_Salary'] = np.log(full_gray['salary_usd'])
    max_exp = full_gray['years_experience'].max()
    full_gray['Experience_Scaled'] = full_gray['years_experience'] / max_exp

    # 行业虚拟变量
    industry_dummies = pd.get_dummies(full_gray['industry'], prefix='Industry')
    full_gray = pd.concat([full_gray, industry_dummies], axis=1)

    # 选择最终列
    final_columns = [
        'salary_usd', 'Log_Salary', 'years_experience', 'Experience_Scaled',
        'AI_Skill_Requirement'
    ] + [col for col in full_gray.columns if col.startswith('Industry_')]

    df_final = full_gray[final_columns].copy()
    df_final.to_csv('/Users/oscar/Desktop/STAT1016GP/version2/data/cleaned_ai_job_data_v2.csv', index=False)

    print(f"\n最终数据统计:")
    print(df_final['AI_Skill_Requirement'].value_counts())
    print(f"\nAI Skill Required: {df_final['AI_Skill_Requirement'].sum()} ({df_final['AI_Skill_Requirement'].mean()*100:.1f}%)")
    print(f"Non-AI: {(1-df_final['AI_Skill_Requirement']).sum()} ({(1-df_final['AI_Skill_Requirement'].mean())*100:.1f}%)")

if __name__ == "__main__":
    main()
