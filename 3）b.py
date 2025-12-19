import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
INPUT_EXCEL = r"C:\Users\64991\Desktop\BEHAVIOURAL FINANCE\ecnometric\group project\trading strategy.xlsx"
OUTPUT_EXCEL = r"C:\Users\64991\Desktop\Fusion_Strategy_2000_Results.xlsx"
OUTPUT_PLOT = r"C:\Users\64991\Desktop\Cumulative_Returns_2000_Onwards.png"

# Load and prepare data (from 2000-07)
df = pd.read_excel(INPUT_EXCEL, header=None, names=['date', 'log_ret'])
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').loc["2000-07-01":]
df['simple_ret'] = np.exp(df['log_ret']) - 1

# Strategy calculations
df['signal_lag1'] = np.where(df['simple_ret'].shift(1) > 0, 1, -1)
df['signal_lag3'] = np.where(df['simple_ret'].shift(3) > 0, 1, -1)
df['fusion_ret'] = (df['signal_lag1'] * 0.5 + df['signal_lag3'] * 0.5) * df['simple_ret']
df['bh_ret'] = df['simple_ret']
df['lag1_ret'] = df['signal_lag1'] * df['simple_ret']
df.fillna(0, inplace=True)

# Cumulative returns
df['cum_fusion'] = (1 + df['fusion_ret']).cumprod() / (1 + df['fusion_ret'].iloc[0])
df['cum_bh'] = (1 + df['bh_ret']).cumprod() / (1 + df['bh_ret'].iloc[0])
df['cum_lag1'] = (1 + df['lag1_ret']).cumprod() / (1 + df['lag1_ret'].iloc[0])


# Backtest metrics calculation
def calc_metrics(ret_series, freq='monthly'):
    n = len(ret_series)
    if n == 0:
        return [np.nan] * 5

    total_ret = ((1 + ret_series).prod() - 1) * 100
    periods_per_year = 12 if freq == 'monthly' else 252
    annual_ret = ((1 + total_ret / 100) ** (periods_per_year / n) - 1) * 100
    sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(periods_per_year) if ret_series.std() != 0 else np.nan
    cum_ret = (1 + ret_series).cumprod()
    max_dd = ((cum_ret / cum_ret.cummax()) - 1).min() * 100
    win_rate = (ret_series > 0).mean() * 100

    return [total_ret, annual_ret, sharpe, max_dd, win_rate]


# Compute stats for all strategies
stats = {
    'Fusion Strategy (50% Lag1+Lag3)': calc_metrics(df['fusion_ret']),
    'Buy-and-Hold': calc_metrics(df['bh_ret']),
    'Pure Lag1': calc_metrics(df['lag1_ret'])
}

backtest_stats = pd.DataFrame({
    'Strategy': stats.keys(),
    'Total_Return_(%)': [s[0] for s in stats.values()],
    'Annual_Return_(%)': [s[1] for s in stats.values()],
    'Sharpe_Ratio': [s[2] for s in stats.values()],
    'Max_Drawdown_(%)': [s[3] for s in stats.values()],
    'Win_Rate_(%)': [s[4] for s in stats.values()]
})

# Plot cumulative returns
plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['cum_fusion'], linewidth=2.5, color='#1f77b4', label='Fusion Strategy')
plt.plot(df.index, df['cum_bh'], linewidth=2, color='#ff7f0e', label='Buy-and-Hold')
plt.plot(df.index, df['cum_lag1'], linewidth=2, color='#2ca02c', label='Pure Lag1')
plt.title('Cumulative Returns (2000.07 Onwards)')
plt.xlabel('Date')
plt.ylabel('Normalized Return (Initial = 1)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

# Save results to Excel
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    backtest_stats.to_excel(writer, sheet_name='Backtest_Stats', index=False)
    excel_data = df.reset_index()[
        ['date', 'simple_ret', 'signal_lag1', 'signal_lag3',
         'fusion_ret', 'bh_ret', 'lag1_ret',
         'cum_fusion', 'cum_bh', 'cum_lag1']
    ]
    excel_data.columns = [
        'Date', 'Index_Return', 'Lag1_Signal', 'Lag3_Signal',
        'Fusion_Return', 'Buy_Hold_Return', 'Pure_Lag1_Return',
        'Cumulative_Fusion', 'Cumulative_Buy_Hold', 'Cumulative_Pure_Lag1'
    ]
    excel_data.to_excel(writer, sheet_name='Monthly_Data', index=False)

# Print summary
print("=== Strategy Execution Completed ===")
print(f"Data Period: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
print(f"Valid Months: {len(df)}")
print("\nBacktest Statistics Summary:")
print(backtest_stats.round(2))
print(f"\nExcel Saved: {OUTPUT_EXCEL}")
print(f"Plot Saved: {OUTPUT_PLOT}")