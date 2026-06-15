Here is the structure for your new loading and analysis notebook. 

It uses the same dynamic, OS-independent path resolution to find your `data/processed` folder, safely loads all five `.parquet` files, and prints out a quick preview of each DataFrame to verify the shapes and structures.

---

### **Cell 1: Setup & Path Configuration**
```python
# %% [markdown]
# # 1. Workspace Setup & Path Configuration
# This cell establishes dynamic paths to easily locate the processed datasets.

# %%
%load_ext autoreload
%autoreload 2

from pathlib import Path
import pandas as pd

# Dynamically find the current directory
try:
    # If running as a standard .py script
    current_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback if running inside a Jupyter Notebook (.ipynb)
    current_dir = Path.cwd()

# Define the path to the processed data folder
# (Goes up to 'stocks', then down into 'data/processed')
processed_dir = current_dir.parent / "data" / "processed"

print(f"Processed data directory resolved to:\n{processed_dir}")
```

---

### **Cell 2: Load the Processed Datasets**
```python
# %% [markdown]
# # 2. Load Processed Datasets
# Loads the five generated `.parquet` files and prints their shapes as a success indicator.

# %%
# Define individual file paths
features_df_path = processed_dir / "features_df.parquet"
macro_df_path = processed_dir / "macro_df.parquet"
df_close_wide_path = processed_dir / "df_close_wide.parquet"
df_atrp_wide_path = processed_dir / "df_atrp_wide.parquet"
df_trp_wide_path = processed_dir / "df_trp_wide.parquet"

# --- Safe Loading with existence checks ---

# 1. Load multi-index feature DataFrame
if features_df_path.exists():
    features_df = pd.read_parquet(features_df_path)
    print(f"[OK] Loaded features_df | Shape: {features_df.shape}")
else:
    print(f"[ERROR] File not found: {features_df_path}")

# 2. Load macro indicator DataFrame
if macro_df_path.exists():
    macro_df = pd.read_parquet(macro_df_path)
    print(f"[OK] Loaded macro_df    | Shape: {macro_df.shape}")
else:
    print(f"[ERROR] File not found: {macro_df_path}")

# 3. Load wide-format close prices
if df_close_wide_path.exists():
    df_close_wide = pd.read_parquet(df_close_wide_path)
    print(f"[OK] Loaded df_close_wide| Shape: {df_close_wide.shape}")
else:
    print(f"[ERROR] File not found: {df_close_wide_path}")

# 4. Load wide-format ATRP (Average True Range Percentage)
if df_atrp_wide_path.exists():
    df_atrp_wide = pd.read_parquet(df_atrp_wide_path)
    print(f"[OK] Loaded df_atrp_wide | Shape: {df_atrp_wide.shape}")
else:
    print(f"[ERROR] File not found: {df_atrp_wide_path}")

# 5. Load wide-format TRP (True Range Percentage)
if df_trp_wide_path.exists():
    df_trp_wide = pd.read_parquet(df_trp_wide_path)
    print(f"[OK] Loaded df_trp_wide  | Shape: {df_trp_wide.shape}")
else:
    print(f"[ERROR] File not found: {df_trp_wide_path}")
```

---

### **Cell 3: Fast Verification & Previews**
```python
# %% [markdown]
# # 3. Quick Data Verification
# Displays previews to inspect the index alignment and ensure the data structures are intact.

# %%
print("\n" + "="*50)
print("DATAFRAME STRUCTURAL PREVIEWS")
print("="*50)

# Preview wide matrix prices (Top-Left 5x5 corner)
if 'df_close_wide' in locals():
    print("\n--- df_close_wide (Prices: First 5 days, First 5 tickers) ---")
    display(df_close_wide.iloc[:5, :5])

# Preview multi-index features
if 'features_df' in locals():
    print("\n--- features_df (First 3 rows of multi-index) ---")
    display(features_df.head(3))

# Preview macro data
if 'macro_df' in locals():
    print("\n--- macro_df (First 3 rows) ---")
    display(macro_df.head(3))
```