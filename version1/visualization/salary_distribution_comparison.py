"""
Salary Distribution: Before and After Log Transformation
Histogram with Normal Distribution Curve Overlay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_csv('/Users/oscar/Downloads/STAT1016GP/data/cleaned_ai_job_data.csv')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 12

# ============================================================
# Create figure with two subplots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ============================================================
# Plot 1: Original Salary Distribution (Right-Skewed)
# ============================================================
ax1 = axes[0]

# Histogram
n1, bins1, patches1 = ax1.hist(
    df['salary_usd'] / 1000,  # Convert to thousands
    bins=50,
    density=True,
    alpha=0.7,
    color='#FF6B6B',
    edgecolor='white',
    linewidth=0.5
)

# Fit and plot normal curve
mu1, std1 = stats.norm.fit(df['salary_usd'] / 1000)
x1 = np.linspace(df['salary_usd'].min() / 1000, df['salary_usd'].max() / 1000, 100)
pdf1 = stats.norm.pdf(x1, mu1, std1)
ax1.plot(x1, pdf1, 'k-', linewidth=2, label=f'Normal Fit\n(μ={mu1:.1f}, σ={std1:.1f})')

# Add skewness annotation
skew1 = stats.skew(df['salary_usd'] / 1000)
ax1.annotate(f'Skewness: {skew1:.2f}\n(Right-skewed)',
             xy=(0.75, 0.85), xycoords='axes fraction',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_title('Before Log Transformation\n(Original Salary in USD)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Salary (Thousands USD)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)

# ============================================================
# Plot 2: Log Salary Distribution (Approximately Normal)
# ============================================================
ax2 = axes[1]

# Histogram
n2, bins2, patches2 = ax2.hist(
    df['Log_Salary'],
    bins=50,
    density=True,
    alpha=0.7,
    color='#4ECDC4',
    edgecolor='white',
    linewidth=0.5
)

# Fit and plot normal curve
mu2, std2 = stats.norm.fit(df['Log_Salary'])
x2 = np.linspace(df['Log_Salary'].min(), df['Log_Salary'].max(), 100)
pdf2 = stats.norm.pdf(x2, mu2, std2)
ax2.plot(x2, pdf2, 'k-', linewidth=2, label=f'Normal Curve\n(μ={mu2:.2f}, σ={std2:.2f})')

# Add skewness annotation
skew2 = stats.skew(df['Log_Salary'])
ax2.annotate(f'Skewness: {skew2:.2f}\n(Approximately Normal)',
             xy=(0.05, 0.85), xycoords='axes fraction',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax2.set_title('After Log Transformation\n(Log(Salary))', fontsize=14, fontweight='bold')
ax2.set_xlabel('Log(Salary)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)

# Add sample size
fig.text(0.5, 0.02, f'n = {len(df):,} observations', ha='center', fontsize=11, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('/Users/oscar/Downloads/STAT1016GP/visualization/salary_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Histogram saved!")
print(f"\nOriginal Salary (USD):")
print(f"  Mean: ${df['salary_usd'].mean():,.0f}")
print(f"  Median: ${df['salary_usd'].median():,.0f}")
print(f"  Skewness: {skew1:.4f}")

print(f"\nLog(Salary):")
print(f"  Mean: {df['Log_Salary'].mean():.4f}")
print(f"  Median: {df['Log_Salary'].median():.4f}")
print(f"  Skewness: {skew2:.4f}")
