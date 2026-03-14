"""
Multiple Linear Regression (MLR) Analysis
Target: log(Salary) = β0 + β1(AI_Skill) + β2(Experience) + β3(Industry) + ε
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load cleaned data
df = pd.read_csv('/Users/oscar/Downloads/STAT1016GP/cleaned_ai_job_data.csv')

# ============================================================
# Prepare Variables
# ============================================================

# Dependent variable
y = df['Log_Salary']

# Independent variables: AI_Skill_Requirement + Experience_Scaled + Industry dummies
# Get all industry dummy columns
industry_cols = [col for col in df.columns if col.startswith('Industry_')]
X_cols = ['AI_Skill_Requirement', 'Experience_Scaled'] + industry_cols

# Convert boolean columns to int
X = df[X_cols].astype(float)

# Add constant (intercept)
X = sm.add_constant(X)

# ============================================================
# Run OLS Regression
# ============================================================
print("="*70)
print("MULTIPLE LINEAR REGRESSION RESULTS")
print("="*70)
print(f"\nDependent Variable: Log_Salary")
print(f"Independent Variables: {len(X.columns) - 1} predictors")
print(f"Sample Size: {len(y)}")
print("-"*70)

model = sm.OLS(y, X)
results = model.fit()

# Print full summary
print(results.summary())

# ============================================================
# Key Metrics Extraction
# ============================================================
print("\n" + "="*70)
print("KEY METRICS SUMMARY")
print("="*70)

# R-squared
r_squared = results.rsquared
r_squared_adj = results.rsquared_adj

print(f"\n📊 MODEL FIT:")
print(f"   R-squared:          {r_squared:.4f} ({r_squared*100:.2f}%)")
print(f"   Adjusted R-squared: {r_squared_adj:.4f}")

# Coefficients
print(f"\n📈 COEFFICIENTS:")
print(f"   β0 (Intercept):     {results.params['const']:.4f}")
print(f"   β1 (AI_Skill):      {results.params['AI_Skill_Requirement']:.4f}")
print(f"   β2 (Experience):    {results.params['Experience_Scaled']:.4f}")

# AI Skill Premium Calculation
ai_coef = results.params['AI_Skill_Requirement']
ai_premium_pct = (np.exp(ai_coef) - 1) * 100

print(f"\n🎯 AI SKILL PREMIUM (Key Finding):")
print(f"   Raw Coefficient (β1):    {ai_coef:.4f}")
print(f"   Exponentiated (e^β1 - 1): {np.exp(ai_coef) - 1:.4f}")
print(f"   AI Skill Premium:          {ai_premium_pct:.2f}%")
print(f"   (Interpretation: Jobs requiring AI skills pay {ai_premium_pct:.2f}% "
      f"more, after controlling for experience and industry)")

# Statistical Significance
print(f"\n📉 STATISTICAL SIGNIFICANCE (P-values):")
print(f"   AI_Skill_Requirement:  P = {results.pvalues['AI_Skill_Requirement']:.4f}")
print(f"   Experience_Scaled:      P = {results.pvalues['Experience_Scaled']:.6f}")

# Check significance
if results.pvalues['AI_Skill_Requirement'] < 0.05:
    print(f"\n   ✅ AI Skill premium IS statistically significant (p < 0.05)")
else:
    print(f"\n   ❌ AI Skill premium is NOT statistically significant (p >= 0.05)")

# Industry coefficients
print(f"\n🏭 INDUSTRY EFFECTS (vs baseline):")
for col in industry_cols:
    coef = results.params[col]
    pval = results.pvalues[col]
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
    print(f"   {col:30s}: {coef:7.4f} (p={pval:.4f}) {sig}")

# ============================================================
# Save results to file
# ============================================================
with open('/Users/oscar/Downloads/STAT1016GP/regression_results.txt', 'w') as f:
    f.write(str(results.summary()))
    f.write("\n\n" + "="*70 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write(f"R-squared: {r_squared:.4f}\n")
    f.write(f"AI Skill Premium: {ai_premium_pct:.2f}%\n")
    f.write(f"P-value (AI_Skill): {results.pvalues['AI_Skill_Requirement']:.4f}\n")

print("\n" + "="*70)
print("Results saved to regression_results.txt")
print("="*70)
