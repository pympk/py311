import os
import subprocess

# 1. Define your run parameters
lookbacks = [10, 20]
os.environ["CACHE_START_DATE"] = "2026-01-01"  # Stays the same for all runs

# 2. Define your execution sequence
scripts = ["0_prepare_data.py", "1_train_model.py", "2_evaluate.py"]

# 3. Loop through each lookback value
for lookback in lookbacks:
    print(f"\n{'#' * 50}")
    print(f"### STARTING PIPELINE WITH CACHE_LOOKBACK = {lookback} ###")
    print(f"{'#' * 50}\n")

    # Update the environment variable for the current run
    # (Must be converted to a string for os.environ)
    os.environ["CACHE_LOOKBACK"] = str(lookback)

    # 4. Run the scripts in order
    for script in scripts:
        print(f"========== Running {script} ==========")

        # Run the script. check=True ensures that if a script crashes,
        # the whole pipeline stops immediately instead of continuing with bad data.
        subprocess.run(["python", script], check=True)

print("\nAll pipeline runs completed successfully!")
