import subprocess
import sys
from pathlib import Path

# --- Define the project's directory structure relative to this script ---
# This script is in the project's root directory.
ROOT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'

# --- List of notebook FILENAMES to run in order ---
# Use the correct, full filenames of your refactored notebooks.
notebooks_to_run = [
    "py0_get_yloader_OHLCV_data_v1.ipynb",
    "py1_clean_df_finviz_v14.ipynb",
    "py2_clean_df_OHLCV_v10.ipynb",
    "py2_save_df_adj_close_v1.ipynb",    
    "py3_calc_perf_ratios_v16.ipynb",
    "py4_append_ratios_v9.ipynb",
    "py5_append_columns_v8.ipynb",
    "py6_append_stats_history_v4.ipynb",
    "py6_view_market_sentiment_history.ipynb",
    "py7_view_daily_market_snapshot.ipynb",
    "py8_portf_picks_short_term_v6.ipynb",
    "py9_backtest_v3.ipynb",
    "py10_backtest_verification_v1.ipynb",
    "py90_view_backtest_results_v5.ipynb",
]


def run_notebook(notebook_path: Path):
    """Executes a notebook using nbconvert and saves it to an 'executed' subdirectory."""
    if not notebook_path.exists():
        print(f"Error: Notebook file not found at {notebook_path}")
        return False

    executed_dir = notebook_path.parent / "executed"
    executed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = executed_dir / f"executed_{notebook_path.name}"
    
    # Use sys.executable to ensure we use the python from the correct virtual env
    command = [
        sys.executable,
        "-m", "jupyter",  # A more robust way to call jupyter
        "nbconvert",
        "--to", "notebook",
        "--execute",
        "--output", str(output_path),
        str(notebook_path)
    ]
    
    print(f"\nRunning command: {' '.join(command)}")
    # Set the working directory for the subprocess to the notebook's directory
    process = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=False, 
        cwd=notebook_path.parent #<-- CRITICAL: Run from the notebook's own directory
    )

    if process.returncode != 0:
        print(f"Error executing {notebook_path.name}:")
        print("--- STDOUT ---")
        print(process.stdout)
        print("--- STDERR ---")
        print(process.stderr)
        return False
    else:
        print(f"Successfully executed {notebook_path.name}")
        print(f"Output saved to: {output_path}")
        return True

# --- Main Execution ---
print("Starting notebook execution sequence...")
for nb_filename in notebooks_to_run:
    # --- FIX: Build the full path to the notebook ---
    full_notebook_path = NOTEBOOKS_DIR / nb_filename
    
    print(f"\n--- Running {nb_filename} ---")
    if not run_notebook(full_notebook_path):
        print(f"Execution failed for {nb_filename}. Stopping sequence.")
        sys.exit(1)

print("\n--- All notebooks executed successfully! ---")