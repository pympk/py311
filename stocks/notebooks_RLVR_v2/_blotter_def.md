"raw_log_reward": log_reward  

    get from main_v3 Holding-Group_Gain  

"actual_return": raw_sleeve_return : 

    slippage_applied = 0.001
    simple_reward = np.exp(raw_log_reward) - 1
    actual_return = simple_reward - slippage_applied

"mkt_return": mkt_return  

    log_return = main_v3 Holding-Benchmark_Gain  

    my_item = "SPY"  
    decision_date = "2022-04-01"
    df = df_ohlcv.loc[my_item].loc[decision_date:].head(7).copy()
    log_return = np.log(df["Adj Close"].iloc[-1] / df["Adj Close"].iloc[1])
    mkt_return = np.exp(log_return) - 1
    print(my_item)
    print(df)
    print(f"\ndf_log_return: {log_return:.6f}")
    print(f"mkt_return: {mkt_return:.6f}")  

"alpha": alpha  

    alpha = actual_return - mkt_return
    print(f"alpha:  {alpha:.6f}")  

"penalized_alpha": penalized_alpha,  

    # Penalize underperformance aggressively for the RL Agent
    penalized_alpha = alpha * self.config.downside_penalty if alpha < 0 else alpha  

"portfolio_impact": portfolio_impact  

    # Math scales it down by holding period logic per capital deployment
    portfolio_impact = raw_sleeve_return / self.holding_period  

"alpha_impact": alpha_impact  

    alpha_impact = alpha / self.holding_period

"agent_equity": self.equity_curve[-1]   

    self.equity_curve = [1.0]  
    self.equity_curve.append(self.equity_curve[-1] * (1.0 + portfolio_impact))  
    agent_equity = self.equity_curve[-1]

"alpha_equity": self.alpha_equity_curve[-1]  

    self.alpha_equity_curve = [1.0]  
    self.alpha_equity_curve.append(self.alpha_equity_curve[-1] * (1.0 + alpha_impact))  


"Rank Offset Percentile": offset  

    universe_size = len(ensemble)
    # Offset is bounded by a percentage of today's available universe
    max_allowed_offset = int(universe_size * rank_max_offset_percentile)

    # Interpolate offset from [0, max_allowed_offset]
    offset = int(np.interp(action[-2], [-1, 1], [0, max_allowed_offset]))

"Rank Width": width

    # Interpolate width from [0, rank_max_width]
    width = int(np.interp(action[-1], [-1, 1], [0, rank_max_width]))


