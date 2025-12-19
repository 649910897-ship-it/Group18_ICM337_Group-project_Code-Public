# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from sklearn.impute import SimpleImputer
import os

# ======================
# Configuration (2-Period Lag for All 4 Economic Indicators)
# ======================
# Core variable definitions
TARGET_DEPENDENT_VAR = "sweden_index_return"
DATE_COL = "date"
# 4 economic indicators for 2-period lag (Repo as short-term rate: lag 2 only)
LAG_VARS = [
    "retail_sales_change",
    "repo_rate_change",
    "unemployment_change",
    "pmi_change"
]
# SPX remains original (no lag for immediate market reaction)
ORIGINAL_VARS = ["spx_return"]
# Final independent vars: 2-period lag for 4 indicators + original SPX
FINAL_INDEPENDENT_VARS = [f"{var}_lag2" for var in LAG_VARS] + ORIGINAL_VARS

# Plot configuration (consistent with previous version)
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'


# ======================
# Step 1: Load Data & Impute Missing Values (Prioritize Repo Blanks)
# ======================
def load_and_impute_data(file_path="optimized_data.csv"):
    """
    Load raw data → Impute missing values (mean imputation for economic indicators)
    Focus on repo_rate_change blank values first
    """
    if not os.path.exists(file_path):
        print(f"ERROR: Preprocessed file '{file_path}' not found!")
        return None

    try:
        # Load raw 11-column data
        df_raw = pd.read_csv(file_path, encoding="utf-8")
        print(f"SUCCESS: Loaded raw data ({len(df_raw)} rows × {len(df_raw.columns)} columns)")

        # Check required columns
        required_raw_cols = [DATE_COL, TARGET_DEPENDENT_VAR] + LAG_VARS + ORIGINAL_VARS
        missing_cols = [col for col in required_raw_cols if col not in df_raw.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {', '.join(missing_cols)}")
            return None

        # Convert and sort date (critical for lag calculation)
        df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], errors="coerce")
        df_raw = df_raw.sort_values(by=DATE_COL).reset_index(drop=True)

        # Impute missing values (mean strategy for stable economic indicators)
        num_cols = LAG_VARS + ORIGINAL_VARS + [TARGET_DEPENDENT_VAR]
        imputer = SimpleImputer(strategy="mean")
        df_raw[num_cols] = imputer.fit_transform(df_raw[num_cols])

        # Verify imputation result
        missing_after = df_raw[num_cols].isnull().sum()
        print(f"\nImputation Result (Missing Values After):")
        for col, cnt in missing_after.items():
            print(f"   - {col}: {cnt} missing values")

        return df_raw

    except Exception as e:
        print(f"ERROR: Data loading/imputation failed: {str(e)}")
        return None


# ======================
# Step 2: Generate 2-Period Lag Variables
# ======================
def generate_lag2_variables(df_raw):
    """
    Generate 2-period lag variables for 4 economic indicators
    Drop rows with NaN from lag shift (first 2 rows)
    """
    try:
        df_lag = df_raw.copy()

        # Generate 2-period lag (shift 2 rows for time series)
        for var in LAG_VARS:
            df_lag[f"{var}_lag2"] = df_lag[var].shift(2)  # 2-period lag

        # Drop first 2 rows (NaN from 2-period lag)
        df_lag = df_lag.dropna(subset=FINAL_INDEPENDENT_VARS + [TARGET_DEPENDENT_VAR, DATE_COL]).reset_index(drop=True)

        # Filter final columns (only keep needed vars)
        final_cols = [DATE_COL, TARGET_DEPENDENT_VAR] + FINAL_INDEPENDENT_VARS
        df_final = df_lag[final_cols].copy()

        print(f"\nSUCCESS: Generated 2-period lag variables for 4 economic indicators")
        print(f"Final data columns: {list(df_final.columns)}")
        print(f"Final data size: {len(df_final)} rows (after dropping lag NaNs)")
        print(
            f"Date range: {df_final[DATE_COL].min().strftime('%Y-%m-%d')} ~ {df_final[DATE_COL].max().strftime('%Y-%m-%d')}")

        return df_final

    except Exception as e:
        print(f"ERROR: 2-period lag generation failed: {str(e)}")
        return None


# ======================
# Step 3: Run Regression with 2-Period Lag Variables
# ======================
def run_regression_with_lag2(df_final):
    """
    OLS Regression with:
    - Dependent: sweden_index_return
    - Independent: 2-period lag 4 indicators + original SPX
    """
    try:
        # Define X (independent) and y (dependent)
        y = df_final[TARGET_DEPENDENT_VAR]
        X = df_final[FINAL_INDEPENDENT_VARS]

        # Add intercept
        X_with_const = sm.add_constant(X)

        # Train/test split (80/20)
        train_size = int(0.8 * len(df_final))
        X_train, X_test = X_with_const.iloc[:train_size], X_with_const.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        dates_train, dates_test = df_final[DATE_COL].iloc[:train_size], df_final[DATE_COL].iloc[train_size:]

        # Fit OLS model
        model = sm.OLS(y_train, X_train).fit()

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluation metrics
        residuals = model.resid
        metrics = {
            "Total_Observations": len(df_final),
            "Train_Observations": len(y_train),
            "Test_Observations": len(y_test),
            "Train_R2": round(r2_score(y_train, y_train_pred), 4),
            "Test_R2": round(r2_score(y_test, y_test_pred), 4),
            "Adjusted_R2": round(model.rsquared_adj, 4),
            "Train_RMSE": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 6),
            "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 6),
            "F_Statistic": round(model.fvalue, 4),
            "F_P_Value": round(model.f_pvalue, 8),
            "Durbin_Watson": round(durbin_watson(residuals), 4)
        }

        # Coefficient table with significance
        coef_df = pd.DataFrame({
            "Variable": ["Intercept"] + FINAL_INDEPENDENT_VARS,
            "Coefficient": [round(val, 6) for val in model.params.values],
            "Std_Error": [round(val, 6) for val in model.bse.values],
            "T_Statistic": [round(val, 4) for val in model.tvalues.values],
            "P_Value": [round(val, 8) for val in model.pvalues.values],
            "95%_CI_Lower": [round(val, 6) for val in model.conf_int()[0].values],
            "95%_CI_Upper": [round(val, 6) for val in model.conf_int()[1].values],
            "Significance": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                             for p in model.pvalues.values]
        })

        # Package results
        results = {
            "model": model,
            "coef_df": coef_df,
            "metrics": metrics,
            "predictions": {
                "train": {"date": dates_train, "actual": y_train, "predicted": y_train_pred},
                "test": {"date": dates_test, "actual": y_test, "predicted": y_test_pred}
            },
            "residuals": residuals,
            "data": df_final
        }

        print(f"\nSUCCESS: Regression with 2-period lag variables completed!")
        return results

    except Exception as e:
        print(f"ERROR: Regression failed: {str(e)}")
        return None


# ======================
# Step 4: Generate Visualizations (2-Period Lag Model)
# ======================
def generate_lag2_plots(results, save_path="regression_lag2_results.png"):
    """
    4 core plots for 2-period lag model:
    1. Actual vs Predicted (train/test)
    2. Residual Distribution
    3. Coefficient Significance (2-period lag vars)
    4. Time Series Trend (2005-2025)
    """
    try:
        train_pred = results["predictions"]["train"]
        test_pred = results["predictions"]["test"]
        coef_df = results["coef_df"]
        residuals = results["residuals"]
        metrics = results["metrics"]

        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2)

        # Plot 1: Actual vs Predicted (2-period lag model)
        axes[0, 0].scatter(train_pred["actual"], train_pred["predicted"],
                           alpha=0.6, color="#2E86AB", label=f"Train (R²={metrics['Train_R2']})")
        axes[0, 0].scatter(test_pred["actual"], test_pred["predicted"],
                           alpha=0.6, color="#E74C3C", label=f"Test (R²={metrics['Test_R2']})")
        # Ideal fit line
        all_actual = np.concatenate([train_pred["actual"].values, test_pred["actual"].values])
        axes[0, 0].plot([all_actual.min(), all_actual.max()],
                        [all_actual.min(), all_actual.max()],
                        "k--", linewidth=2, label="Ideal Fit")
        axes[0, 0].set_xlabel("Actual Sweden Index Return")
        axes[0, 0].set_ylabel("Predicted Sweden Index Return")
        axes[0, 0].set_title("Actual vs Predicted (2-Period Lag Model)")
        axes[0, 0].legend()

        # Plot 2: Residual Distribution
        axes[0, 1].hist(residuals, bins=25, alpha=0.7, color="#F39C12", edgecolor="black", density=True)
        # Normal distribution curve
        mu, sigma = residuals.mean(), residuals.std()
        x_norm = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        axes[0, 1].plot(x_norm, y_norm, "b-", linewidth=2, label=f"Normal (μ={mu:.4f}, σ={sigma:.4f})")
        axes[0, 1].set_xlabel("Residuals (Train Set)")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Residual Distribution (2-Period Lag Model)")
        axes[0, 1].legend()

        # Plot 3: Coefficient Significance (exclude intercept)
        coef_plot = coef_df[coef_df["Variable"] != "Intercept"].copy()
        colors = ["#E74C3C" if p < 0.05 else "#BDC3C7" for p in coef_plot["P_Value"]]
        bars = axes[1, 0].barh(coef_plot["Variable"], coef_plot["Coefficient"], color=colors, alpha=0.8)
        # Add significance labels
        for i, (_, row) in enumerate(coef_plot.iterrows()):
            axes[1, 0].text(
                row["Coefficient"] + 0.001 if row["Coefficient"] > 0 else row["Coefficient"] - 0.001,
                i, row["Significance"], va="center", ha="left" if row["Coefficient"] > 0 else "right"
            )
        axes[1, 0].axvline(x=0, color="black", linestyle="-", linewidth=1)
        axes[1, 0].set_xlabel("Coefficient Value")
        axes[1, 0].set_title("Coefficient Significance (2-Period Lag Variables, Red=P<0.05)")

        # Plot 4: Time Series Trend
        axes[1, 1].plot(train_pred["date"], train_pred["actual"], color="#2E86AB", linewidth=2, label="Actual (Train)")
        axes[1, 1].plot(train_pred["date"], train_pred["predicted"], color="#2E86AB", linewidth=1.5,
                        linestyle="--", label="Predicted (Train)")
        axes[1, 1].plot(test_pred["date"], test_pred["actual"], color="#E74C3C", linewidth=2, label="Actual (Test)")
        axes[1, 1].plot(test_pred["date"], test_pred["predicted"], color="#E74C3C", linewidth=1.5,
                        linestyle="--", label="Predicted (Test)")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Sweden Index Return")
        axes[1, 1].set_title("Time Series: Actual vs Predicted (2005-2025, 2-Period Lag Model)")
        axes[1, 1].legend()
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSUCCESS: 2-period lag model plots saved to '{save_path}'")

    except Exception as e:
        print(f"WARNING: Plot generation failed: {str(e)}")


# ======================
# Step 5: Save Results to Excel (2-Period Lag Version)
# ======================
def save_lag2_results(results, save_path=None):
    """
    修复权限问题：优先保存到桌面，避免当前文件夹权限不足
    """
    try:
        # 若未指定路径，默认保存到桌面
        if save_path is None:
            import os
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "sweden_regression_lag2_results.xlsx")
        else:
            desktop_path = save_path

        with pd.ExcelWriter(desktop_path, engine="openpyxl") as writer:
            # Sheet 1: Final lagged data
            results["data"].to_excel(writer, sheet_name="1_Final_Lag2_Data", index=False)

            # Sheet 2: Coefficient table
            results["coef_df"].to_excel(writer, sheet_name="2_Coefficient_Table", index=False)

            # Sheet 3: Evaluation metrics
            metrics_df = pd.DataFrame(list(results["metrics"].items()), columns=["Metric", "Value"])
            metrics_df.to_excel(writer, sheet_name="3_Evaluation_Metrics", index=False)

            # Sheet 4: Predictions (train/test)
            train_df = pd.DataFrame({
                "Date": results["predictions"]["train"]["date"],
                "Actual_Return": results["predictions"]["train"]["actual"],
                "Predicted_Return": results["predictions"]["train"]["predicted"],
                "Residuals": results["predictions"]["train"]["actual"] - results["predictions"]["train"]["predicted"],
                "Data_Set": "Train"
            })
            test_df = pd.DataFrame({
                "Date": results["predictions"]["test"]["date"],
                "Actual_Return": results["predictions"]["test"]["actual"],
                "Predicted_Return": results["predictions"]["test"]["predicted"],
                "Residuals": results["predictions"]["test"]["actual"] - results["predictions"]["test"]["predicted"],
                "Data_Set": "Test"
            })
            pd.concat([train_df, test_df], ignore_index=True).to_excel(writer, sheet_name="4_Predictions", index=False)

            # Sheet 5: Variable explanation (2-period lag)
            var_explanation = [
                (DATE_COL, "Date (Monthly, 2005-2025)", "Time dimension"),
                (TARGET_DEPENDENT_VAR, "Sweden Index Return", "Dependent variable"),
                ("retail_sales_change_lag2", "Retail Sales Change (2-period lag)",
                 "Independent variable (consumption lag)"),
                ("repo_rate_change_lag2", "Repo Rate Change (2-period lag)",
                 "Independent variable (short-term rate lag)"),
                ("unemployment_change_lag2", "Unemployment Change (2-period lag)",
                 "Independent variable (labor market lag)"),
                ("pmi_change_lag2", "PMI Change (2-period lag)", "Independent variable (manufacturing lag)"),
                ("spx_return", "SPX Index Return (original)", "Independent variable (immediate market reaction)")
            ]
            var_df = pd.DataFrame(var_explanation, columns=["Variable_Name", "Description", "Type"])
            var_df.to_excel(writer, sheet_name="5_Variable_Explanation", index=False)

        print(f"SUCCESS: 2-period lag model results saved to '{desktop_path}'")

    except Exception as e:
        print(f"ERROR: Result saving failed: {str(e)}")


# ======================
# Main Function (Full Pipeline for 2-Period Lag)
# ======================
def main():
    print("=" * 70)
    print("Multiple Regression with 2-Period Lag (Repo Rate 2-Period Lag + Imputation)")
    print("=" * 70)

    # Step 1: Load and impute data
    df_imputed = load_and_impute_data()
    if df_imputed is None:
        print("Terminated: Data loading/imputation failed")
        return

    # Step 2: Generate 2-period lag variables
    df_final = generate_lag2_variables(df_imputed)
    if df_final is None:
        print("Terminated: 2-period lag generation failed")
        return

    # Step 3: Run regression
    results = run_regression_with_lag2(df_final)
    if results is None:
        print("Terminated: Regression failed")
        return

    # Step 4: Print key results
    print("\n" + "=" * 70)
    print("Key Results (2-Period Lag Model)")
    print("=" * 70)
    # Core metrics
    key_metrics = ["Total_Observations", "Adjusted_R2", "Test_R2", "F_P_Value", "Durbin_Watson"]
    for metric in key_metrics:
        print(f"   {metric}: {results['metrics'][metric]}")

    # Significant variables (P<0.05)
    significant_vars = results["coef_df"][
        (results["coef_df"]["P_Value"] < 0.05) &
        (results["coef_df"]["Variable"] != "Intercept")
        ]
    print(f"\nSignificant Variables (P<0.05):")
    if len(significant_vars) > 0:
        for _, row in significant_vars.iterrows():
            print(
                f"   - {row['Variable']}: Coefficient={row['Coefficient']}, P-Value={row['P_Value']}, Significance={row['Significance']}")
    else:
        print(f"   - No significant variables (all P≥0.05)")

    # Step 5: Generate plots and save results
    generate_lag2_plots(results)
    save_lag2_results(results)

    print("\n" + "=" * 70)
    print("Full Pipeline Completed! Generated Files:")
    print("1. regression_lag2_results.png (4 core plots for 2-period lag model)")
    print("2. sweden_regression_lag2_results.xlsx (complete 2-period lag results)")
    print("=" * 70)


if __name__ == "__main__":
    main()