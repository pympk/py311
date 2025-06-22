import subprocess
import subprocess
import sys
import os
from pathlib import Path # <--- Import Path from pathlib


# List of notebooks to run in order
notebooks = [
    # "py0_get_yloader_OHLCV_data_v1.ipynb",
    # "py1_clean_df_finviz_v14.ipynb",
    # "py2_clean_df_OHLCV_v10.ipynb",
    # "py2_save_df_adj_close_v1.ipynb",    
    # "py3_calc_perf_ratios_v16.ipynb",
    # "py4_append_ratios_v9.ipynb",
    # "py5_append_columns_v8.ipynb",
    "py6_append_stats_history_v4.ipynb",
    "py6_view_market_sentiment_history.ipynb",
    "py7_view_daily_market_snapshot.ipynb",
    "py8_portf_picks_short_term_v6.ipynb",
    "py9_backtest_v2.ipynb",
    "py10_backtest_verification_v1.ipynb"
    "py90_view_backtest_results_v5.ipynb",
]

def run_notebook(notebook_path):
    """Executes a notebook using nbconvert and saves it to an 'executed' subdirectory."""

    # Create a Path object from the input notebook path
    p_notebook = Path(notebook_path)

    # Define the target directory name
    executed_dir_name = "executed" # Use a string

    # Create the full path to the target directory
    # p_notebook.parent is the directory containing the original notebook
    executed_dir_path = p_notebook.parent / executed_dir_name

    # --- Ensure the target directory exists ---
    # os.makedirs(executed_dir_path, exist_ok=True) # Using os module
    executed_dir_path.mkdir(parents=True, exist_ok=True) # Using pathlib (preferred)

    # # --- Clear the contents of the executed directory ---
    # # Added section to delete existing contents
    # print(f"Clearing contents of directory: {executed_dir_path}")
    # deleted_count = 0
    # error_count = 0
    # for item in executed_dir_path.iterdir():
    #     try:
    #         if item.is_file() or item.is_symlink():
    #             item.unlink() # Deletes files and symbolic links
    #             print(f"  Deleted file: {item.name}")
    #             deleted_count += 1
    #         elif item.is_dir():
    #             shutil.rmtree(item) # Deletes directories and their contents recursively
    #             print(f"  Deleted directory: {item.name}")
    #             deleted_count += 1
    #     except Exception as e:
    #         print(f"  Error deleting {item.name}: {e}")
    #         error_count += 1
    # if deleted_count > 0 or error_count > 0:
    #     print(f"Finished clearing: {deleted_count} items deleted, {error_count} errors.")
    # else:
    #     print("Directory was already empty or contained no deletable items.")
    # # --- End of added section ---

    # Create the new filename by prefixing the original name
    output_filename = f"executed_{p_notebook.name}"

    # Create the new Path object by joining the target directory path
    # with the new filename.
    output_path_obj = executed_dir_path / output_filename # Corrected join

    # Convert the Path object back to a string for use with subprocess
    output_path = str(output_path_obj)

    # Ensure you are using the jupyter from the correct environment,
    # sys.executable often points to the python interpreter.
    # Constructing the path to jupyter might be more robust in complex setups.
    jupyter_executable = os.path.join(os.path.dirname(sys.executable), 'jupyter') # More robust way to find jupyter

    command = [
        jupyter_executable, # Use the specific jupyter executable
        "nbconvert",
        "--to", "notebook",
        "--execute",
        "--output", output_path, # Use the correctly constructed path
        # "--inplace", # Uncomment if you want to modify the original
        # "--allow-errors", # Uncomment to continue on errors
        notebook_path
    ]
    print(f"\nRunning command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False) # check=False allows us to handle errors manually

    if process.returncode != 0:
        print(f"Error executing {notebook_path}:")
        print("--- STDOUT ---")
        print(process.stdout)
        print("--- STDERR ---")
        print(process.stderr)
        return False
    else:
        print(f"Successfully executed {notebook_path}")
        print(f"Output saved to: {output_path}")
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