"""
================================================================================
Independent Samples T-Test Analysis - Version 2
================================================================================
Purpose: Test whether there is a statistically significant difference in
         log(Salary) between jobs requiring AI skills vs. those that do not

Hypothesis:
  H0: μ_with_AI = μ_without_AI (No significant difference in mean salary)
  H1: μ_with_AI ≠ μ_without_AI (Significant difference exists)

Significance Level: α = 0.05

Changes from Version 1:
- Uses cleaned_ai_job_data_v2.csv from new data cleaning pipeline
- Three-layer AI classification (Regex -> Gray Zone -> API)
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Configuration
BASE_DIR = '/Users/oscar/Desktop/STAT1016GP'
VERSION2_DIR = os.path.join(BASE_DIR, 'version2')
DATA_DIR = os.path.join(VERSION2_DIR, 'data')
TTEST_DIR = os.path.join(VERSION2_DIR, 't_test_analysis')

# Output directory
os.makedirs(TTEST_DIR, exist_ok=True)

# ==================== Load Data ====================
print("="*70)
print("Independent Samples T-Test Analysis (Version 2)")
print("="*70)

# Load cleaned data (V2)
data_path = os.path.join(DATA_DIR, 'cleaned_ai_job_data_v2.csv')

# Check if V2 data exists, fallback to V1 if not
if not os.path.exists(data_path):
    print(f"[Warning] V2 data not found at {data_path}")
    print("[Info] Falling back to V1 data...")
    data_path = os.path.join(BASE_DIR, 'data', 'cleaned_ai_job_data.csv')

print(f"\n[Loading Data] {data_path}")
df = pd.read_csv(data_path)
print(f"Data Source: {os.path.basename(data_path)}")

# Split into two groups
salary_ai = df[df['AI_Skill_Requirement'] == 1]['Log_Salary']
salary_no_ai = df[df['AI_Skill_Requirement'] == 0]['Log_Salary']

print(f"\n[Sample Descriptive Statistics]")
print("-"*50)
print(f"With AI Skill Requirement (AI_Skill=1):")
print(f"  Sample Size: {len(salary_ai)}")
print(f"  Mean: {salary_ai.mean():.4f}")
print(f"  Std Dev: {salary_ai.std():.4f}")
print(f"  Median: {salary_ai.median():.4f}")

print(f"\nWithout AI Skill Requirement (AI_Skill=0):")
print(f"  Sample Size: {len(salary_no_ai)}")
print(f"  Mean: {salary_no_ai.mean():.4f}")
print(f"  Std Dev: {salary_no_ai.std():.4f}")
print(f"  Median: {salary_no_ai.median():.4f}")

# ==================== Normality Check ====================
print("\n" + "="*70)
print("[Normality Check]")
print("="*70)
print("Note: Due to large sample size, we rely on Central Limit Theorem")
print("      and assume approximate normal distribution")

# ==================== Levene's Test ====================
print("\n" + "="*70)
print("[Levene's Test for Equality of Variances]")
print("="*70)

levene_stat, levene_p = stats.levene(salary_ai, salary_no_ai)
print(f"Levene Statistic: {levene_stat:.4f}")
print(f"p-value: {levene_p:.6f}")

if levene_p < 0.05:
    print("-> Conclusion: Unequal variances (p < 0.05), using Welch's t-test")
    equal_var = False
else:
    print("-> Conclusion: Equal variances (p >= 0.05), using standard t-test")
    equal_var = True

# ==================== Independent Samples T-Test ====================
print("\n" + "="*70)
print("[Independent Samples T-Test Results]")
print("="*70)

# Perform t-test
t_stat, p_value = stats.ttest_ind(salary_ai, salary_no_ai, equal_var=equal_var)

print(f"\nHypothesis:")
print(f"  H0: μ_with_AI = μ_without_AI (No significant difference)")
print(f"  H1: μ_with_AI ≠ μ_without_AI (Significant difference exists)")
print(f"\nTest Results:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6e}")

# Cohen's d effect size
pooled_std = np.sqrt(((len(salary_ai)-1)*salary_ai.std()**2 + (len(salary_no_ai)-1)*salary_no_ai.std()**2) / (len(salary_ai)+len(salary_no_ai)-2))
cohens_d = (salary_ai.mean() - salary_no_ai.mean()) / pooled_std

print(f"  Cohen's d (Effect Size): {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_size = "Negligible"
elif abs(cohens_d) < 0.5:
    effect_size = "Small"
elif abs(cohens_d) < 0.8:
    effect_size = "Medium"
else:
    effect_size = "Large"
print(f"  Effect Size Interpretation: {effect_size}")

# ==================== Conclusion ====================
print("\n" + "="*70)
print("[Final Conclusion]")
print("="*70)

alpha = 0.05
if p_value < alpha:
    print(f"✓ p-value ({p_value:.6e}) < α ({alpha})")
    print(f"✓ REJECT H0")
    print(f"\n[Conclusion]: There IS a statistically significant difference in salary")
    print(f"              between jobs with and without AI skill requirements")
else:
    print(f"✗ p-value ({p_value:.6e}) >= α ({alpha})")
    print(f"✗ FAIL TO REJECT H0")
    print(f"\n[Conclusion]: The salary difference is NOT statistically significant")

# 95% Confidence Interval for mean difference
mean_diff = salary_ai.mean() - salary_no_ai.mean()
se_diff = pooled_std * np.sqrt(1/len(salary_ai) + 1/len(salary_no_ai))
ci_lower = mean_diff - 1.96 * se_diff
ci_upper = mean_diff + 1.96 * se_diff

print(f"\n[Mean Difference]")
print(f"  Difference (with AI - without AI): {mean_diff:.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Original salary difference (back-transform)
salary_diff = np.exp(salary_ai.mean()) - np.exp(salary_no_ai.mean())
print(f"\n[Original Salary Difference (Back-transformed)]")
print(f"  Mean Salary (with AI): ${np.exp(salary_ai.mean()):,.2f}")
print(f"  Mean Salary (without AI): ${np.exp(salary_no_ai.mean()):,.2f}")
print(f"  Difference: ${salary_diff:,.2f}")
print(f"  Percentage Difference: {(np.exp(mean_diff)-1)*100:.2f}%")

# ==================== Save Results ====================
print("\n" + "="*70)
print("[Save Results]")
print("="*70)

# Save detailed results to text file
results_text = f"""
================================================================================
Independent Samples T-Test Results Report - Version 2
================================================================================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {os.path.basename(data_path)}

One. Sample Descriptive Statistics
--------------------------------------------------------------------------------
With AI Skill Requirement (AI_Skill=1):
  Sample Size: {len(salary_ai)}
  Mean (Log_Salary): {salary_ai.mean():.4f}
  Standard Deviation: {salary_ai.std():.4f}
  Median: {salary_ai.median():.4f}

Without AI Skill Requirement (AI_Skill=0):
  Sample Size: {len(salary_no_ai)}
  Mean (Log_Salary): {salary_no_ai.mean():.4f}
  Standard Deviation: {salary_no_ai.std():.4f}
  Median: {salary_no_ai.median():.4f}

Two. Hypothesis Testing
--------------------------------------------------------------------------------
H0: μ_with_AI = μ_without_AI (No significant difference in salary)
H1: μ_with_AI ≠ μ_without_AI (Significant difference exists)

Significance Level: α = 0.05

Three. Test Results
--------------------------------------------------------------------------------
Levene's Test for Equality of Variances:
  Statistic: {levene_stat:.4f}
  p-value: {levene_p:.6f}
  Conclusion: {'Unequal variances, using Welch\'s t-test' if not equal_var else 'Equal variances assumed, using standard t-test'}

Independent Samples T-Test:
  t-statistic: {t_stat:.4f}
  p-value: {p_value:.6e}
  Cohen's d (Effect Size): {cohens_d:.4f} ({effect_size} Effect)

Mean Difference:
  Difference (Log): {mean_diff:.4f}
  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]

Original Salary Difference:
  Mean Salary (with AI): ${np.exp(salary_ai.mean()):,.2f}
  Mean Salary (without AI): ${np.exp(salary_no_ai.mean()):,.2f}
  Absolute Difference: ${salary_diff:,.2f}
  Percentage Difference: {(np.exp(mean_diff)-1)*100:.2f}%

Four. Final Conclusion
--------------------------------------------------------------------------------
{'✓ REJECT H0: Significant difference exists (p < 0.05)' if p_value < alpha else '✗ FAIL TO REJECT H0: No significant difference (p >= 0.05)'}

================================================================================
"""

results_path = os.path.join(TTEST_DIR, 'ttest_results.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write(results_text)

print(f"✓ Results report saved to: {results_path}")

# Save summary data
summary_df = pd.DataFrame({
    'Group': ['With AI Skill (AI_Skill=1)', 'Without AI Skill (AI_Skill=0)'],
    'Sample_Size': [len(salary_ai), len(salary_no_ai)],
    'Log_Salary_Mean': [salary_ai.mean(), salary_no_ai.mean()],
    'Log_Salary_Std': [salary_ai.std(), salary_no_ai.std()],
    'Original_Salary_Mean': [np.exp(salary_ai.mean()), np.exp(salary_no_ai.mean())]
})

summary_path = os.path.join(TTEST_DIR, 'ttest_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"✓ Summary data saved to: {summary_path}")

print("\n" + "="*70)
print("[Analysis Complete]")
print("="*70)
