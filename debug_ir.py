import pandas as pd
import numpy as np

df = pd.read_csv('stocks/notebooks_RLVR/gemini/IR_63.csv')

# Use the columns from the CSV
tsla_ret = df['TSLA_Ret_1d']
spy_ret = df['SPY_Ret_1d']
act_ret = df['Act_Ret']

window = 63

# Calculation 1: Active Return (TSLA - SPY)
mu_act = act_ret.rolling(window).mean()
sigma_act = act_ret.rolling(window).std()
ir_act = mu_act / sigma_act

# Calculation 2: Absolute Return (TSLA) - Sharpe
mu_tsla = tsla_ret.rolling(window).mean()
sigma_tsla = tsla_ret.rolling(window).std()
ir_tsla = mu_tsla / sigma_tsla

print("Row 64 (index 63):")
print(f"TSLA_IR_63 (from CSV): {df.loc[63, 'TSLA_IR_63']}")
print(f"Excel_IR_63 (from CSV): {df.loc[63, 'Excel_IR_63']}")
print(f"Calculated IR (Active): {ir_act.loc[63]}")
print(f"Calculated IR (Sharpe): {ir_tsla.loc[63]}")

print("\nRow 67 (index 66):")
print(f"TSLA_IR_63 (from CSV): {df.loc[66, 'TSLA_IR_63']}")
print(f"Excel_IR_63 (from CSV): {df.loc[66, 'Excel_IR_63']}")
print(f"Calculated IR (Active): {ir_act.loc[66]}")
print(f"Calculated IR (Sharpe): {ir_tsla.loc[66]}")

# Calculation 7: Log Returns
tsla_log_ret = np.log(df['TSLA_Adj_Close'] / df['TSLA_Adj_Close'].shift(1))
spy_log_ret = np.log(df['SPY_Adj_Close'] / df['SPY_Adj_Close'].shift(1))
act_log_ret = tsla_log_ret - spy_log_ret
print(f'\nCalculated IR (Log Active) at 64: {(act_log_ret.rolling(window).mean() / act_log_ret.rolling(window).std()).loc[63]}')
print(f'\nCalculated IR (Log Sharpe) at 64: {(tsla_log_ret.rolling(window).mean() / tsla_log_ret.rolling(window).std()).loc[63]}')
print(f'\nCalculated IR (Log Active) at 67: {(act_log_ret.rolling(window).mean() / act_log_ret.rolling(window).std()).loc[66]}')
print(f'\nCalculated IR (Log Sharpe) at 67: {(tsla_log_ret.rolling(window).mean() / tsla_log_ret.rolling(window).std()).loc[66]}')
