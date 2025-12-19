# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset, acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ======================
# Core Configuration
# ======================
EXCEL_PATH = "C:\\Users\\64991\\PycharmProjects\\PythonProject\\sweden_regression_lag2_results.xlsx"
SHEET_NAME = "1_Final_Lag2_Data"
TARGET_DEPENDENT = "sweden_index_return"
INDEPENDENT_VARS = [
    "retail_sales_change_lag2",
    "repo_rate_change_lag2",
    "unemployment_change_lag2",
    "pmi_change_lag2",
    "spx_return"
]


# ======================
# Step 1: Load Data from Excel
# ======================
def load_excel_final_data():
    print("=" * 100)
    print("Step 1: Load Final Data from Excel")
    print("=" * 100)
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
        print(f"✅ SUCCESS: Loaded data with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns in Excel: {list(df.columns)}")

        required_cols = [TARGET_DEPENDENT] + INDEPENDENT_VARS
        df_clean = df[required_cols].dropna().reset_index(drop=True)
        print(f"✅ Cleaned data: {len(df_clean)} valid rows (no missing values)")
        return df_clean
    except Exception as e:
        print(f"❌ ERROR: Failed to load data: {str(e)}")
        return None


# ======================
# Step 2: CLRM 5 Assumption Tests
# ======================
def run_clrm_tests(df):
    print("\n" + "=" * 100)
    print("Step 2: CLRM Diagnostic Testing Results (All 5 Assumptions)")
    print("=" * 100)

    y = df[TARGET_DEPENDENT]
    X = df[INDEPENDENT_VARS]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # (a) Homoscedasticity
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, X_with_const)
    print("\n(a) Homoscedasticity (Breusch-Pagan Test, H0: Homoscedasticity)")
    print(f"  - Test Statistic: {bp_stat:.4f}")
    print(f"  - P-Value: {bp_pval:.8f}")
    print(
        f"  - Conclusion: {'❌ Reject H0 → Heteroscedasticity exists (violates CLRM)' if bp_pval < 0.05 else '✅ Fail to reject H0 → Homoscedasticity holds (satisfies CLRM)'}")

    # (b) Linear Functional Form
    reset_test = linear_reset(model, power=2)
    reset_stat = reset_test.statistic
    reset_pval = reset_test.pvalue
    print("\n(b) Linear Functional Form (Ramsey RESET Test, H0: Correct Linear Form)")
    print(f"  - Test Statistic: {reset_stat:.4f}")
    print(f"  - P-Value: {reset_pval:.8f}")
    print(
        f"  - Conclusion: {'❌ Reject H0 → Linear form incorrect (violates CLRM)' if reset_pval < 0.05 else '✅ Fail to reject H0 → Linear form holds (satisfies CLRM)'}")

    # (c) Normality of Errors
    jb_stat, jb_pval, _, _ = jarque_bera(model.resid)
    print("\n(c) Normality of Errors (Jarque-Bera Test, H0: Normal Distribution)")
    print(f"  - Test Statistic: {jb_stat:.4f}")
    print(f"  - P-Value: {jb_pval:.8f}")
    print(
        f"  - Conclusion: {'❌ Reject H0 → Errors not normal (violates CLRM)' if jb_pval < 0.05 else '✅ Fail to reject H0 → Errors are normal (satisfies CLRM)'}")

    # (d) Serial Correlation
    bg_test = acorr_breusch_godfrey(model, nlags=1)
    bg_stat, bg_pval = bg_test[0], bg_test[1]
    print("\n(d) Serial Correlation (Breusch-Godfrey Test, Lag=1, H0: No Serial Correlation)")
    print(f"  - Test Statistic: {bg_stat:.4f}")
    print(f"  - P-Value: {bg_pval:.8f}")
    print(
        f"  - Conclusion: {'❌ Reject H0 → Serial correlation exists (violates CLRM)' if bg_pval < 0.05 else '✅ Fail to reject H0 → No serial correlation (satisfies CLRM)'}")

    # (e) Multicollinearity
    vif_data = pd.DataFrame({
        "Variable": INDEPENDENT_VARS,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    max_vif = vif_data["VIF"].max()
    print("\n(e) Multicollinearity (VIF Test, Threshold: VIF>10=Severe)")
    for _, row in vif_data.iterrows():
        print(f"  - {row['Variable']}: VIF={row['VIF']:.4f}")
    print(f"  - Max VIF: {max_vif:.4f}")
    if max_vif > 10:
        print(f"  - Conclusion: ❌ Severe multicollinearity (violates CLRM)")
    elif max_vif > 5:
        print(f"  - Conclusion: ⚠️ Moderate multicollinearity (weakly violates CLRM)")
    else:
        print(f"  - Conclusion: ✅ No significant multicollinearity (satisfies CLRM)")

    return model  # Return model for subsequent statistical output


# ======================
# Step 3: Output Full Statistical Results
# ======================
def output_full_stats(model, df):
    print("\n" + "=" * 100)
    print("Step 3: Full Statistical Results (Model + Coefficients + Residuals)")
    print("=" * 100)
    y = df[TARGET_DEPENDENT]
    X = df[INDEPENDENT_VARS]
    X_with_const = sm.add_constant(X)
    y_pred = model.predict(X_with_const)
    residuals = model.resid

    # 1. Model Fit
    print("\n📊 1. Model Fit Statistics")
    print("-" * 50)
    print(f"  - R-Squared (R²): {model.rsquared:.4f}")
    print(f"  - Adjusted R-Squared: {model.rsquared_adj:.4f}")
    print(f"  - F-Statistic: {model.fvalue:.4f}")
    print(f"  - F-Statistic P-Value: {model.f_pvalue:.8f}")
    print(f"  - AIC: {model.aic:.4f}")
    print(f"  - BIC: {model.bic:.4f}")
    print(f"  - Log-Likelihood: {model.llf:.4f}")

    # 2. Coefficients
    print("\n📈 2. Coefficient Estimates & Significance")
    print("-" * 50)
    coef_df = pd.DataFrame({
        "Variable": ["Intercept"] + INDEPENDENT_VARS,
        "Coefficient": [round(val, 6) for val in model.params.values],
        "Std_Error": [round(val, 6) for val in model.bse.values],
        "T-Statistic": [round(val, 4) for val in model.tvalues.values],
        "P-Value": [round(val, 8) for val in model.pvalues.values],
        "95%_CI_Lower": [round(val, 6) for val in model.conf_int()[0].values],
        "95%_CI_Upper": [round(val, 6) for val in model.conf_int()[1].values],
        "Significance": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                         for p in model.pvalues.values]
    })
    for _, row in coef_df.iterrows():
        print(f"  - {row['Variable']}:")
        print(f"    Coef={row['Coefficient']}, StdErr={row['Std_Error']}, T={row['T-Statistic']}, P={row['P-Value']}")
        print(f"    95% CI: [{row['95%_CI_Lower']}, {row['95%_CI_Upper']}], Sig={row['Significance']}")

    # 3. Residual Stats
    print("\n📉 3. Residual Statistics")
    print("-" * 50)
    print(f"  - Mean of Residuals: {residuals.mean():.6f}")
    print(f"  - Std Dev of Residuals: {residuals.std():.6f}")
    print(f"  - Skewness: {residuals.skew():.4f}")
    print(f"  - Kurtosis: {residuals.kurtosis():.4f}")
    print(f"  - Min Residual: {residuals.min():.6f}")
    print(f"  - Max Residual: {residuals.max():.6f}")
    print(f"  - Durbin-Watson: {sm.stats.stattools.durbin_watson(residuals):.4f}")

    # 4. Prediction Performance
    print("\n🎯 4. Prediction Performance")
    print("-" * 50)
    mse = np.mean((y - y_pred) **2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100 if y.min() != 0 else np.nan
    print(f"  - Mean Squared Error (MSE): {mse:.6f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.4f}%" if not np.isnan(
        mape) else "  - MAPE: N/A (Avoid division by zero)")


# ======================
# Step 4: Export All Results to Excel
# ======================
def export_results_to_excel(model, df, excel_path="CLRM_Full_Results.xlsx"):
    print("\n" + "=" * 100)
    print("Step 4: Export All Results to Excel")
    print("=" * 100)

    y = df[TARGET_DEPENDENT]
    X = df[INDEPENDENT_VARS]
    X_with_const = sm.add_constant(X)
    y_pred = model.predict(X_with_const)
    residuals = model.resid

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # ----------------------
            # Sheet 1: CLRM Test Results
            # ----------------------
            bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, X_with_const)
            reset_test = linear_reset(model, power=2)
            jb_stat, jb_pval, _, _ = jarque_bera(model.resid)
            bg_test = acorr_breusch_godfrey(model, nlags=1)
            vif_data = pd.DataFrame({
                "Variable": INDEPENDENT_VARS,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            })

            clrm_results = pd.DataFrame({
                "Test_Item": [
                    "Homoscedasticity (Breusch-Pagan)",
                    "Linear Functional Form (Ramsey RESET)",
                    "Normality of Errors (Jarque-Bera)",
                    "Serial Correlation (Breusch-Godfrey, Lag=1)",
                    "Multicollinearity (Max VIF)"
                ],
                "Test_Statistic": [bp_stat, reset_test.statistic, jb_stat, bg_test[0], vif_data["VIF"].max()],
                "P_Value": [bp_pval, reset_test.pvalue, jb_pval, bg_test[1], None],
                "Conclusion": [
                    "Fail to reject H0 → Homoscedasticity holds" if bp_pval >= 0.05 else "Reject H0 → Heteroscedasticity exists",
                    "Fail to reject H0 → Linear form holds" if reset_test.pvalue >= 0.05 else "Reject H0 → Linear form incorrect",
                    "Fail to reject H0 → Errors are normal" if jb_pval >= 0.05 else "Reject H0 → Errors not normal",
                    "Fail to reject H0 → No serial correlation" if bg_test[
                                                                       1] >= 0.05 else "Reject H0 → Serial correlation exists",
                    "No significant multicollinearity" if vif_data[
                                                              "VIF"].max() <= 5 else "Moderate/Severe multicollinearity"
                ],
                "CLRM_Compliance": [
                    "Yes" if bp_pval >= 0.05 else "No",
                    "Yes" if reset_test.pvalue >= 0.05 else "No",
                    "Yes" if jb_pval >= 0.05 else "No",
                    "Yes" if bg_test[1] >= 0.05 else "No",
                    "Yes" if vif_data["VIF"].max() <= 5 else "No"
                ]
            })
            clrm_results.to_excel(writer, sheet_name="1_CLRM_Test_Results", index=False)

            # ----------------------
            # Sheet 2: Model Fit Statistics
            # ----------------------
            fit_stats = pd.DataFrame({
                "Statistic_Name": [
                    "R-Squared (R²)",
                    "Adjusted R-Squared",
                    "F-Statistic",
                    "F-Statistic P-Value",
                    "AIC",
                    "BIC",
                    "Log-Likelihood",
                    "Total_Observations"
                ],
                "Value": [
                    model.rsquared,
                    model.rsquared_adj,
                    model.fvalue,
                    model.f_pvalue,
                    model.aic,
                    model.bic,
                    model.llf,
                    len(df)
                ],
                "Format": ["%.4f", "%.4f", "%.4f", "%.8f", "%.4f", "%.4f", "%.4f", "%.0f"]
            })
            for idx, row in fit_stats.iterrows():
                fit_stats.at[idx, "Formatted_Value"] = row["Format"] % row["Value"]
            fit_stats = fit_stats[["Statistic_Name", "Value", "Formatted_Value"]]
            fit_stats.to_excel(writer, sheet_name="2_Model_Fit_Stats", index=False)

            # ----------------------
            # Sheet 3: Coefficient Details
            # ----------------------
            coef_df = pd.DataFrame({
                "Variable": ["Intercept"] + INDEPENDENT_VARS,
                "Coefficient": model.params.values,
                "Std_Error": model.bse.values,
                "T_Statistic": model.tvalues.values,
                "P_Value": model.pvalues.values,
                "95%_CI_Lower": model.conf_int()[0].values,
                "95%_CI_Upper": model.conf_int()[1].values,
                "Significance": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                                 for p in model.pvalues.values]
            })
            coef_df[["Coefficient", "Std_Error", "T_Statistic", "P_Value", "95%_CI_Lower", "95%_CI_Upper"]] = \
                coef_df[["Coefficient", "Std_Error", "T_Statistic", "P_Value", "95%_CI_Lower", "95%_CI_Upper"]].round(6)
            coef_df.to_excel(writer, sheet_name="3_Coefficient_Details", index=False)

            # ----------------------
            # Sheet 4: Residual Statistics
            # ----------------------
            residual_stats = pd.DataFrame({
                "Statistic_Name": [
                    "Mean of Residuals",
                    "Std Dev of Residuals",
                    "Skewness",
                    "Kurtosis",
                    "Minimum Residual",
                    "Maximum Residual",
                    "Durbin-Watson Statistic"
                ],
                "Value": [
                    residuals.mean(),
                    residuals.std(),
                    residuals.skew(),
                    residuals.kurtosis(),
                    residuals.min(),
                    residuals.max(),
                    sm.stats.stattools.durbin_watson(residuals)
                ]
            }).round(6)
            residual_stats.to_excel(writer, sheet_name="4_Residual_Stats", index=False)

            # ----------------------
            # Sheet 5: Prediction Metrics
            # ----------------------
            mse = np.mean((y - y_pred)** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))
            mape = np.mean(np.abs((y - y_pred) / y)) * 100 if y.min() != 0 else np.nan

            prediction_metrics = pd.DataFrame({
                "Metric_Name": ["MSE", "RMSE", "MAE", "MAPE (%)"],
                "Value": [mse, rmse, mae, mape]
            }).round(6)
            prediction_metrics.to_excel(writer, sheet_name="5_Prediction_Metrics", index=False)

            # ----------------------
            # Sheet 6: Raw Data + Predictions + Residuals
            # ----------------------
            full_data = df.copy()
            full_data["Predicted_Return"] = y_pred.round(6)
            full_data["Residuals"] = residuals.round(6)
            full_data.to_excel(writer, sheet_name="6_Raw_Data_Predictions", index=False)

            # ----------------------
            # Sheet 7: VIF Details
            # ----------------------
            vif_data = vif_data.round(4)
            vif_data["Conclusion"] = ["No multicollinearity" if vif <= 5 else "Multicollinearity exists"
                                      for vif in vif_data["VIF"]]
            vif_data.to_excel(writer, sheet_name="7_VIF_Details", index=False)

        print(f"✅ Success: All results exported to '{excel_path}'")
        print(f"📋 Excel contains 7 worksheets:")
        print(
            "  1. CLRM Test Results  2. Model Fit Statistics  3. Coefficient Details  4. Residual Statistics  5. Prediction Metrics  6. Raw Data + Predictions  7. VIF Details")
    except Exception as e:
        print(f"❌ Error: Failed to export to Excel: {str(e)}")
        print("  Recommendation: Close the Excel file with the same name, or modify the export path")


# ======================
# Main Function
# ======================
def main():
    df = load_excel_final_data()
    if df is None:
        print("\n❌ Terminated: Data loading failed")
        return

    model = run_clrm_tests(df)
    output_full_stats(model, df)
    export_results_to_excel(model, df)  # New: Export to Excel

    print("\n" + "=" * 100)
    print("✅ Full Analysis & Export Completed! Results Ready for Use.")
    print("=" * 100)


if __name__ == "__main__":
    main()