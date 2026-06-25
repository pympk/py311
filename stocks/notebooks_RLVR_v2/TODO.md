refactor 3 rlvr files
common utils: clean df_ohlcv, load dataframes, path names, master_calendar
do we used these df_atrp_wide, df_trp_wide 


add Slippage 5-basis-point slippage penalty 
Stack the 189d and 21d Feature Cubes
Reward about SPY


create data dir
move and save data from data dir
use output dir only for output 
verify alphacache


add date prefix to RL outputs
save interim cache
write to config in part 1
pandas to polars


Goal is to write a verification notebook that independently verify feature_cube's calculation (see features_cube.info()). For example, user inputs Date = "2026-01-02", Ticker = "A", the verification notebook will independently calculate feature_cube's output for verification. 

The codebase is large. Be precise in the chat. Ask if user is not clear. Ask if you need to see more code. Don't assume. It will save tokens and keep context window clear in the long run. 

here is RLVR_Part1_AlphaCache_v2.ipynb, the code that produce feature_cube.
