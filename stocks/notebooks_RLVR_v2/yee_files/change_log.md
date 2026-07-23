2026-07-20  
-- Saved features_df, macro_df, df_ohlcv, df_fed, and df_indices to LOCAL_DATA_DIR. The RLVR training, validation, and testing environments will remain frozen and in perfect sync with the generated AlphaCache under LOCAL_DATA_DIR  

2026-07-18  
-- Replaced clip with np.arcsinh(scaled_x) in transform method inside class ObservationScaler, adapter.py  
-- Added update_lr, a linearly decays the learning rate from its initial value down to 0 over the course of training, trainer.py PPOTrainer   
-- Changed 02 run parameters
-- Added test_verify_oos_returns.py  