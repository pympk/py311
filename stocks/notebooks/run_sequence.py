import subprocess
import subprocess
import sys
import os
from pathlib import Path # <--- Import Path from pathlib

# List of notebooks to run in order
notebooks = [
    "py1_clean_df_finviz_v7.ipynb",
    "py2_clean_df_OHLCV_v5.ipynb",
    "py3_calc_perf_ratios_v10.ipynb",
    "py4_append_ratios_v7.ipynb",
    "py5_append_columns_v5.ipynb",
    "py6_stats_history_v1.ipynb",
    "py7_cov_corr_matrices_v4.ipynb",
    "py8_portf_picks_v31.ipynb",
    "py9_market_sentiment.ipynb_v1
    "py10_append_selection_v0.ipynb",
]

def run_notebook(notebook_path):
    """Executes a notebook using nbconvert."""

    # --- Start Modification ---
    # Create a Path object from the input notebook path
    p_notebook = Path(notebook_path)

    # Create the new filename by prefixing the original name
    output_filename = f"executed_{p_notebook.name}"

    # Create the new Path object by joining the original parent directory
    # with the new filename. The '/' operator handles joining paths.
    output_path_obj = p_notebook.parent / output_filename

    # Convert the Path object back to a string for use with subprocess
    output_path = str(output_path_obj)
    # --- End Modification ---
    # Ensure you are using the jupyter from the correct environment,
    # sys.executable often points to the python interpreter.
    # Constructing the path to jupyter might be more robust in complex setups.
    jupyter_executable = os.path.join(os.path.dirname(sys.executable), 'jupyter') # More robust way to find jupyter

    command = [
        jupyter_executable, # Use the specific jupyter executable
        "nbconvert",
        "--to", "notebook",
        "--execute",
        "--output", output_path,
        # "--inplace", # Uncomment if you want to modify the original
        # "--allow-errors", # Uncomment to continue on errors
        notebook_path
    ]
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True)

    if process.returncode != 0:
        print(f"Error executing {notebook_path}:")
        print(process.stdout)
        print(process.stderr)
        return False
    else:
        print(f"Successfully executed {notebook_path}")
        # Optional: print stdout even on success if needed
        # print(process.stdout)
        return True

# --- Main Execution ---
print("Starting notebook execution sequence...")
for nb in notebooks:
    print(f"\n--- Running {nb} ---")
    if not run_notebook(nb):
        print(f"Execution failed for {nb}. Stopping sequence.")
        sys.exit(1) # Exit with error code

print("\n--- All notebooks executed successfully! ---")