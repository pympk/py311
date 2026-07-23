import pandas as pd
from core.paths import GLOBAL_DATA_DIR, GLOBAL_PROCESSED_DIR, LOCAL_DATA_DIR


def load_raw_global_data():
    """Loads raw OHLCV, Fed, and Indices from the global data folder and downcasts them."""
    df_ohlcv, df_fed, df_indices = None, None, None

    ohlcv_path = GLOBAL_DATA_DIR / "df_OHLCV_stocks_etfs.parquet"
    if ohlcv_path.exists():
        df_ohlcv = pd.read_parquet(ohlcv_path)
        # REMOVE OR COMMENT THESE OUT float32:
        # price_cols = ["Adj Open", "Adj High", "Adj Low", "Adj Close"]
        # df_ohlcv[price_cols] = df_ohlcv[price_cols].astype("float32")
        # df_ohlcv["Volume"] = df_ohlcv["Volume"].astype("float32")

    fed_path = GLOBAL_DATA_DIR / "High_Yield_Spread_T10Y2Y_Spread.csv"
    if fed_path.exists():
        df_fed = pd.read_csv(fed_path)
        if "Unnamed: 0" in df_fed.columns:
            df_fed = df_fed.rename(columns={"Unnamed: 0": "Date"})
        df_fed[["High_Yield_Spread", "Yield_Curve_10Y2Y"]] = df_fed[
            ["High_Yield_Spread", "Yield_Curve_10Y2Y"]
        ].astype("float32")

    indices_path = GLOBAL_DATA_DIR / "df_indices.parquet"
    if indices_path.exists():
        df_indices = pd.read_parquet(indices_path)
        df_indices[["Adj Open", "Adj High", "Adj Low", "Adj Close"]] = df_indices[
            ["Adj Open", "Adj High", "Adj Low", "Adj Close"]
        ].astype("float32")
        df_indices["Volume"] = df_indices["Volume"].astype("float32")

    return df_ohlcv, df_fed, df_indices


def load_processed_data():
    """Loads aligned and preprocessed data needed for Cache building and Training."""
    # ---> REDIRECTED TO LOCAL_DATA_DIR TO PREVENT DESYNC WITH ALPHACACHE <---
    df_ohlcv = pd.read_parquet(LOCAL_DATA_DIR / "df_ohlcv.parquet")
    macro_df = pd.read_parquet(LOCAL_DATA_DIR / "macro_df.parquet")
    features_df = pd.read_parquet(LOCAL_DATA_DIR / "features_df.parquet")
    return df_ohlcv, macro_df, features_df
