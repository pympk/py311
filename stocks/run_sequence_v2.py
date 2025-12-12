# run_sequence_v1.py

import subprocess
import sys
from pathlib import Path

# =============================================================================
# === 1. SCRIPT CONFIGURATION
# =============================================================================

# --- Define the project's directory structure relative to this script ---
# This robustly determines the project's root directory by finding the
# location of this script (`__file__`) and getting its parent folder.
# This ensures the script works correctly regardless of where it's called from.
ROOT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT_DIR / 'notebooks_mean_reversion'

# --- Master Control List for the Pipeline ---
# This list defines the exact sequence of notebooks to be executed.
# The order is critical, as each notebook often depends on the output of the previous one.
# Use the full, correct filenames of your refactored notebooks.
notebooks_to_run = [
    "py1_clean_df_finviz_v15.ipynb",
    "py2_clean_df_OHLCV_v12.ipynb",
    "py2_save_df_adj_close_v2.ipynb",    
    "py3_calc_perf_ratios_v17.ipynb",
    "py4_append_ratios_v10.ipynb",
    "py5_append_columns_v8.ipynb",
    "py6_append_stats_history_v4.ipynb",
    "py6_view_market_sentiment_history_v1.ipynb",
    "py7_view_daily_market_snapshot_v0.ipynb",
    "py8_portf_picks_short_term_v6.ipynb",
    "py9_backtest_v3.ipynb",
    "py10_backtest_verification_v3.ipynb",
    "py90_interactive_backtest_v0.ipynb",  # Executed notebook won't display chart, run manually
]



# =============================================================================
# === 2. CORE EXECUTION FUNCTION
# =============================================================================

def run_notebook(notebook_path: Path) -> bool:
    """
    Executes a given Jupyter notebook using nbconvert and saves the output.

    This function handles the complexities of calling a command-line tool from Python,
    including path management, error capturing, and setting the correct working directory.

    Args:
        notebook_path (Path): A pathlib.Path object pointing to the notebook to be executed.

    Returns:
        bool: True if the notebook executed successfully, False otherwise.
    """
    # --- Guard Clause: Check if the notebook file actually exists ---
    if not notebook_path.exists():
        print(f"Error: Notebook file not found at {notebook_path}")
        return False

    # --- Output Path Management: Keep executed notebooks organized ---
    # Create a subdirectory named 'executed' inside the notebooks folder.
    # This prevents cluttering the main directory with output files.
    executed_dir = notebook_path.parent / "executed"
    executed_dir.mkdir(parents=True, exist_ok=True) # `exist_ok=True` prevents errors if the dir already exists.
    
    # Define a clear output filename, e.g., "executed_py1_clean_df_finviz_v14.ipynb".
    output_path = executed_dir / f"executed_{notebook_path.name}"
    
    # --- Command Construction: Build the command to run nbconvert ---
    # This list will be passed to the subprocess.
    command = [
        sys.executable,        # Use the Python executable from the current environment.
        "-m", "jupyter",       # A robust way to call jupyter, ensuring it's from the same env.
        "nbconvert",
        "--to", "notebook",
        "--execute",           # This flag tells nbconvert to run all cells in the notebook.
        "--output", str(output_path),  # Specify the full output path.
        str(notebook_path)     # The input notebook to run.
    ]
    
    print(f"\nRunning command: {' '.join(command)}")
    
    # --- Subprocess Execution: Run the command and capture its output ---
    # The `cwd` argument is CRITICAL. It sets the working directory for the command.
    # We set it to the notebook's own folder, which ensures that any relative paths
    # inside the notebook (like importing `config.py` from the root) work correctly.
    # `check=False` prevents the script from crashing on error, so we can handle it.
    process = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=False, 
        cwd=notebook_path.parent 
    )

    # --- Error Handling: Check if the notebook ran successfully ---
    if process.returncode != 0:
        print(f"Error executing {notebook_path.name}:")
        print("--- STDOUT ---")
        print(process.stdout)
        print("--- STDERR ---")
        print(process.stderr)
        return False  # Signal failure
    else:
        print(f"Successfully executed {notebook_path.name}")
        print(f"Output saved to: {output_path}")
        return True  # Signal success

# =============================================================================
# === 3. MAIN WORKFLOW
# =============================================================================

print("Starting notebook execution sequence...")

# Loop through the master list of notebooks defined at the top.
for nb_filename in notebooks_to_run:
    
    # Build the full, unambiguous path to the notebook.
    full_notebook_path = NOTEBOOKS_DIR / nb_filename
    
    print(f"\n--- Running {nb_filename} ---")
    
    # --- Fail-Fast Logic: Stop the entire sequence if any notebook fails ---
    # This is essential for automation. If a step like data cleaning fails,
    # we should not proceed to the analysis steps with bad data.
    if not run_notebook(full_notebook_path):
        print(f"Execution failed for {nb_filename}. Stopping sequence.")
        sys.exit(1)  # Exit the script with a non-zero status code to indicate an error.

print("\n--- All notebooks executed successfully! ---")