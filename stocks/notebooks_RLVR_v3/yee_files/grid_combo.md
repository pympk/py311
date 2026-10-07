1 GRID COMBO = 1 set of hyperparameters
    └── For each SEED (e.g., 987654):
            └── Spawn 64 ENVS, each starting at a RANDOM date in Cal Train
                    └── Each env steps forward 256 days (NUM_STEPS) chronologically
                            └── Collect 16,384 experiences
                                    └── Update agent weights (PPO, 4 passes, mini-batches of 4096)
                                            └── Every 3 epochs (EVAL_FREQ), test on Cal Val
                                                    └── If Val Sharpe best → Save model
                                                    └── If no improvement for 15 checks (PATIENCE) → STOP