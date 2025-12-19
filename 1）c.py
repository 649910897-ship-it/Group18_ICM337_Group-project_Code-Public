import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.gofplots import qqplot

# ---------------------- Data Preparation & Cleaning ----------------------
file_path = r"C:\Users\64991\Desktop\BEHAVIOURAL FINANCE\ecnometric\group project\report\return rate and statistics(Sweden and Sri lanka).xlsx"
sweden_df = pd.read_excel(file_path, sheet_name="Sweden", skiprows=3)

raw_data = {
    "Dates": sweden_df.iloc[:, 0],
    "Return": sweden_df.iloc[:, 3]
}

df = pd.DataFrame(raw_data)
df["Dates"] = pd.to_datetime(df["Dates"], format="%d/%m/%Y", errors="coerce")
df["Return"] = pd.to_numeric(df["Return"], errors="coerce")
df["Return_Lag1"] = df["Return"].shift(1)
df_clean = df.dropna(subset=["Dates", "Return"]).reset_index(drop=True)
df_clean = df_clean[(df_clean["Dates"] >= "2005-07-01") & (df_clean["Dates"] <= "2025-06-30")]

# ---------------------- OLS Regression Function ----------------------
def run_ols_regression(data):
    valid_data = data.dropna(subset=["Return_Lag1"])
    X_valid = sm.add_constant(valid_data["Return_Lag1"])
    y_valid = valid_data["Return"]
    model = sm.OLS(y_valid, X_valid).fit()
    result = {
        "Sample_Size": len(valid_data),
        "Intercept(α)": model.params["const"],
        "Lag1_Coeff(β)": model.params["Return_Lag1"],
        "Lag1_Pvalue": model.pvalues["Return_Lag1"],
        "Lag1_Tstat": model.tvalues["Return_Lag1"],
        "R_Squared": model.rsquared,
        "Adjusted_R2": model.rsquared_adj,
        "Durbin_Watson": durbin_watson(model.resid)
    }
    return result, model

# ---------------------- Run Regression & Normality Tests ----------------------
full_result, model = run_ols_regression(df_clean)
residuals = model.resid  # Extract residuals for normality test

# 1. Jarque-Bera Normality Test
jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)
full_result["JB_Statistic"] = jb_stat
full_result["JB_Pvalue"] = jb_pvalue

# 2. Normality Visualization (Histogram + Q-Q Plot)
desktop_path = r"C:\Users\64991\Desktop"
plot_path = f"{desktop_path}\\Sweden_AR1_Normality_Plots.png"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram with normal curve
ax1.hist(residuals, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax1.plot(x, (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2)), 'r--', linewidth=2)
ax1.set_title('Residual Histogram vs Normal Distribution')
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Density')

# Q-Q Plot
qqplot(residuals, line='s', ax=ax2)
ax2.set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

# ---------------------- Print Results (Including Normality) ----------------------
print("=== Data Overview ===")
print(f"Time Range: {df_clean['Dates'].min().strftime('%Y-%m')} to {df_clean['Dates'].max().strftime('%Y-%m')}")
print(f"Valid Observations: {len(df_clean)}")

print("\n===== Sweden MSCI AR(1) Regression Results =====")
print(f"Sample Size: {full_result['Sample_Size']:.0f}")
print(f"Intercept (α): {full_result['Intercept(α)']:.4f}")
print(f"1-Month Lagged Coeff (β): {full_result['Lag1_Coeff(β)']:.4f} (P-Value: {full_result['Lag1_Pvalue']:.4f})")
print(f"R²: {full_result['R_Squared']:.4f} | Adjusted R²: {full_result['Adjusted_R2']:.4f}")
print(f"Durbin-Watson: {full_result['Durbin_Watson']:.4f}")

print("\n===== Residual Normality Test (Jarque-Bera) =====")
print(f"JB Statistic: {full_result['JB_Statistic']:.4f}")
print(f"JB P-Value: {full_result['JB_Pvalue']:.4f}")
print("Conclusion: " + ("Reject H0 (Not Normal)" if full_result['JB_Pvalue'] < 0.05 else "Fail to Reject H0 (Normal)"))

# ---------------------- Export Results to Excel ----------------------
output_file = f"{desktop_path}\\Sweden_AR1_Results.xlsx"
result_df = pd.DataFrame({
    "Metric": [
        "Sample Size", "Intercept (α)", "1-Month Lagged Coeff (β)",
        "β P-Value", "β T-Statistic", "R²", "Adjusted R²",
        "Durbin-Watson", "JB Statistic", "JB P-Value"
    ],
    "Value": [
        full_result['Sample_Size'],
        full_result['Intercept(α)'],
        full_result['Lag1_Coeff(β)'],
        full_result['Lag1_Pvalue'],
        full_result['Lag1_Tstat'],
        full_result['R_Squared'],
        full_result['Adjusted_R2'],
        full_result['Durbin_Watson'],
        full_result['JB_Statistic'],
        full_result['JB_Pvalue']
    ]
})

result_df.to_excel(output_file, index=False, engine="openpyxl")
print(f"\n✅ Results exported to: {output_file}")
print(f"✅ Normality plots saved to: {plot_path}")