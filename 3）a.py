import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

# ---------------------- Data Preparation & Cleaning (Original Logic) ----------------------
# Replace with your actual file path (MANDATORY)
file_path = r"C:\Users\64991\Desktop\BEHAVIOURAL FINANCE\ecnometric\group project\report\return rate and statistics(Sweden and Sri lanka).xlsx"

# Read Sweden MSCI data (original skip 3 rows)
sweden_df = pd.read_excel(file_path, sheet_name="Sweden", skiprows=3)

# Extract core columns (original: Date=col0, Return=col2)
raw_data = {
    "Dates": sweden_df.iloc[:, 0],
    "Return": sweden_df.iloc[:, 3]
}

# Convert to DataFrame and clean (original logic)
df = pd.DataFrame(raw_data)
df["Dates"] = pd.to_datetime(df["Dates"], format="%d/%m/%Y", errors="coerce")
df["Return"] = pd.to_numeric(df["Return"], errors="coerce")
df["Return_Lag1"] = df["Return"].shift(1)  # 1-month lagged return
df_clean = df.dropna(subset=["Dates", "Return"]).reset_index(drop=True)
df_clean = df_clean[(df_clean["Dates"] >= "2005-07-01") & (df_clean["Dates"] <= "2025-06-30")]

# ---------------------- OLS Regression Function (Original) ----------------------
def run_ols_regression(data):
    X = sm.add_constant(data["Return_Lag1"])
    y = data["Return"]
    # Filter valid data for regression (original)
    valid_data = data.dropna(subset=["Return_Lag1"])
    X_valid = sm.add_constant(valid_data["Return_Lag1"])
    y_valid = valid_data["Return"]
    model = sm.OLS(y_valid, X_valid).fit()
    # Extract original results
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

# ---------------------- Run Full Sample Regression (NO SUBPERIODS) ----------------------
full_result, _ = run_ols_regression(df_clean)

# ---------------------- Print Original Results (English Output) ----------------------
print("=== Data Overview (Original) ===")
print(f"Time Range: {df_clean['Dates'].min().strftime('%Y-%m')} to {df_clean['Dates'].max().strftime('%Y-%m')}")
print(f"Valid Observations: {len(df_clean)}")

print("\n===== Sweden MSCI AR(1) Regression Results (Full Sample, 2005-2025) =====")
print(f"Sample Size: {full_result['Sample_Size']:.0f}")
print(f"Intercept (α): {full_result['Intercept(α)']:.4f}")
print(f"1-Month Lagged Return Coefficient (β): {full_result['Lag1_Coeff(β)']:.4f}")
print(f"β P-Value: {full_result['Lag1_Pvalue']:.4f}")
print(f"β T-Statistic: {full_result['Lag1_Tstat']:.4f}")
print(f"R²: {full_result['R_Squared']:.4f}")
print(f"Adjusted R²: {full_result['Adjusted_R2']:.4f}")
print(f"Durbin-Watson Statistic: {full_result['Durbin_Watson']:.4f}")

desktop_path = r"C:\Users\64991\Desktop"  # Your desktop absolute path
output_file = f"{desktop_path}\\Sweden_AR1_Results.xlsx"  # Export file path

# Organize results into DataFrame
result_df = pd.DataFrame({
    "Metric": [
        "Sample Size", "Intercept (α)", "1-Month Lagged Coeff (β)",
        "β P-Value", "β T-Statistic", "R²", "Adjusted R²", "Durbin-Watson"
    ],
    "Value": [
        full_result['Sample_Size'],
        full_result['Intercept(α)'],
        full_result['Lag1_Coeff(β)'],
        full_result['Lag1_Pvalue'],
        full_result['Lag1_Tstat'],
        full_result['R_Squared'],
        full_result['Adjusted_R2'],
        full_result['Durbin_Watson']
    ]
})

# Export to desktop Excel
result_df.to_excel(output_file, index=False, engine="openpyxl")
print(f"\n✅ Results exported to: {output_file}")