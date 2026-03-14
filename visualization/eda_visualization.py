"""
EDA Visualization for AI Job Market Salary Analysis
- Correlation Heatmap
- Boxplot comparison by AI Skill Requirement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Load cleaned data
df = pd.read_csv('/Users/oscar/Downloads/STAT1016GP/cleaned_ai_job_data.csv')

# ============================================================
# 1. CORRELATION HEATMAP
# ============================================================
# Select relevant columns for correlation
corr_columns = ['Log_Salary', 'Experience_Scaled', 'AI_Skill_Requirement']
df_corr = df[corr_columns]

# Calculate correlation matrix
corr_matrix = df_corr.corr()

# Create figure
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Create heatmap with annotations
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.4f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
    annot_kws={'size': 14, 'fontweight': 'bold'},
    ax=ax1
)

# Customize
ax1.set_title('Correlation Heatmap\nLog_Salary, Experience_Scaled, AI_Skill_Requirement',
              fontsize=16, fontweight='bold', pad=20)

# Adjust labels
labels = ['Log(Salary)', 'Experience (Scaled)', 'AI Skill Required']
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.set_yticklabels(labels, rotation=0)

plt.tight_layout()
plt.savefig('/Users/oscar/Downloads/STAT1016GP/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Correlation Heatmap saved!")
print("\nCorrelation Matrix:")
print(corr_matrix)

# ============================================================
# 2. BOXPLOT - Salary Distribution by AI Skill Requirement
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Prepare data
df['AI_Skill_Label'] = df['AI_Skill_Requirement'].map({1: 'AI Skill Required (1)', 0: 'No AI Skill Required (0)'})

# Create boxplot
box_colors = ['#FF6B6B', '#4ECDC4']
bp = sns.boxplot(
    x='AI_Skill_Label',
    y='Log_Salary',
    data=df,
    palette=box_colors,
    width=0.5,
    ax=ax2
)

# Add individual data points
sns.stripplot(
    x='AI_Skill_Label',
    y='Log_Salary',
    data=df,
    color='black',
    alpha=0.3,
    size=1,
    ax=ax2
)

# Calculate statistics for annotation
stats_dict = df.groupby('AI_Skill_Requirement')['Log_Salary'].agg(['median', 'mean', 'std', 'count'])
print("\nBoxplot Statistics:")
print(stats_dict)

# Add median labels
medians = df.groupby('AI_Skill_Requirement')['Log_Salary'].median()
for i, (idx, median) in enumerate(medians.items()):
    ax2.annotate(f'Median: {median:.3f}',
                xy=(i, median),
                xytext=(i + 0.25, median + 0.1),
                fontsize=11,
                fontweight='bold',
                color='darkblue')

# Customize
ax2.set_title('Salary Distribution Comparison\nby AI Skill Requirement', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('AI Skill Requirement', fontsize=13, fontweight='bold')
ax2.set_ylabel('Log(Salary)', fontsize=13, fontweight='bold')
ax2.set_ylim(10, 13)

# Add grid
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
ax2.set_axisbelow(True)

# Add legend for sample sizes
n0 = len(df[df['AI_Skill_Requirement'] == 0])
n1 = len(df[df['AI_Skill_Requirement'] == 1])
ax2.text(0.02, 0.98, f'n (No AI) = {n0}\nn (AI Required) = {n1}',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/oscar/Downloads/STAT1016GP/boxplot_salary_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nBoxplot saved!")
print(f"Sample sizes - No AI Skill: {n0}, AI Skill Required: {n1}")

# ============================================================
# Additional Summary Statistics
# ============================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal observations: {len(df)}")
print(f"\nLog_Salary range: {df['Log_Salary'].min():.3f} - {df['Log_Salary'].max():.3f}")
print(f"Experience range: {df['years_experience'].min()} - {df['years_experience'].max()} years")
print(f"\nAI Skill Requirement distribution:")
print(df['AI_Skill_Requirement'].value_counts())
