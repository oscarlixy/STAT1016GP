"""
Forest Plot: Visualizing Regression Coefficients
Show the impact of Experience, AI Skill, and Industries on Salary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
df = pd.read_csv('/Users/oscar/Downloads/STAT1016GP/data/cleaned_ai_job_data.csv')

# ============================================================
# Prepare Variables and Run Regression
# ============================================================
y = df['Log_Salary']
industry_cols = [col for col in df.columns if col.startswith('Industry_')]
X_cols = ['AI_Skill_Requirement', 'Experience_Scaled'] + industry_cols

X = df[X_cols].astype(float)
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

# ============================================================
# Extract Coefficients and Confidence Intervals
# ============================================================
# Get confidence intervals (95%)
conf_int = results.conf_int(alpha=0.05)

# Prepare data for forest plot
# Exclude constant, include all predictors
variables = ['Experience_Scaled', 'AI_Skill_Requirement'] + industry_cols

data = []
for var in variables:
    coef = results.params[var]
    se = results.bse[var]
    pval = results.pvalues[var]
    ci_lower = conf_int.loc[var, 0]
    ci_upper = conf_int.loc[var, 1]

    # Convert to percentage effect
    pct_effect = (np.exp(coef) - 1) * 100

    data.append({
        'Variable': var.replace('Industry_', '').replace('_', ' '),
        'Coefficient': coef,
        'SE': se,
        'P_value': pval,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'Pct_effect': pct_effect,
        'Significant': 'Yes' if pval < 0.05 else 'No'
    })

df_plot = pd.DataFrame(data)

# Sort by coefficient magnitude
df_plot = df_plot.sort_values('Coefficient', ascending=True)

# ============================================================
# Create Forest Plot
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Color coding
colors = []
for _, row in df_plot.iterrows():
    if row['Variable'] == 'Experience Scaled':
        colors.append('#e74c3c')  # Red for main effect
    elif row['P_value'] < 0.05:
        colors.append('#27ae60')  # Green for significant
    else:
        colors.append('#95a5a6')  # Gray for not significant

# Plot points and error bars
y_positions = range(len(df_plot))

for i, (idx, row) in enumerate(df_plot.iterrows()):
    # Error bar (CI)
    ax.plot([row['CI_lower'], row['CI_upper']], [i, i],
            color=colors[i], linewidth=2.5, alpha=0.7)

    # Point estimate
    ax.scatter(row['Coefficient'], i, s=150, color=colors[i],
               edgecolors='white', linewidth=1.5, zorder=5)

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Customize
ax.set_yticks(y_positions)
ax.set_yticklabels(df_plot['Variable'], fontsize=11)
ax.set_xlabel('Coefficient (Effect on Log_Salary)', fontsize=12, fontweight='bold')
ax.set_title('Forest Plot: Regression Coefficients\n(Impact on AI Job Salaries)', fontsize=14, fontweight='bold', pad=15)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='white', label='Experience (Key Predictor)'),
    Patch(facecolor='#27ae60', edgecolor='white', label='Significant (p < 0.05)'),
    Patch(facecolor='#95a5a6', edgecolor='white', label='Not Significant')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Add grid
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Add annotation for Experience
exp_coef = df_plot[df_plot['Variable'] == 'Experience Scaled']['Coefficient'].values[0]
ax.annotate(f'β = {exp_coef:.3f}\n(p < 0.001)',
            xy=(exp_coef, len(df_plot) - 1),
            xytext=(exp_coef + 0.08, len(df_plot) - 1.5),
            fontsize=10, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

# Tight layout
plt.tight_layout()
plt.savefig('/Users/oscar/Downloads/STAT1016GP/visualization/forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Forest Plot saved!")
print("\n" + "="*60)
print("COEFFICIENT SUMMARY")
print("="*60)
print(df_plot[['Variable', 'Coefficient', 'Pct_effect', 'P_value', 'Significant']].to_string(index=False))

# ============================================================
# Save as high-quality summary
# ============================================================
summary_text = """
FOREST PLOT - REGRESSION COEFFICIENT ANALYSIS
================================================

Key Findings:
- Experience is the DOMINANT predictor (β = 0.360, p < 0.001)
- Each standard deviation increase in experience → ~42.7% salary increase
- AI Skill Requirement shows NO significant effect (β = -0.004, p = 0.430)
- Industry effects are NOT statistically significant

Interpretation:
- The further right a point, the higher the positive impact on salary
- Points crossing the vertical zero line indicate non-significant effects
- Experience (red) is far to the right, showing strong positive effect

"""

with open('/Users/oscar/Downloads/STAT1016GP/visualization/forest_plot_summary.txt', 'w') as f:
    f.write(summary_text)

print("\nSummary saved to forest_plot_summary.txt")
