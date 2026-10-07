import pandas as pd
from core.paths import GLOBAL_DATA_DIR, LOCAL_DATA_DIR
from core.contracts import ProcessedDataBundle


def load_raw_global_data():
    """Loads raw OHLCV, Fed, and Indices from the global data folder and downcasts them."""
    df_ohlcv, df_fed, df_indices = None, None, None

    ohlcv_path = GLOBAL_DATA_DIR / "df_OHLCV_stocks_etfs.parquet"
    if ohlcv_path.exists():
        df_ohlcv = pd.read_parquet(ohlcv_path).reset_index()
        # ROOT LEVEL FIX: Force timezone-naive, normalized to midnight EOD dates
        df_ohlcv["Date"] = (
            pd.to_datetime(df_ohlcv["Date"], utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )
        df_ohlcv = df_ohlcv.drop_duplicates(subset=["Ticker", "Date"], keep="last")
        df_ohlcv = df_ohlcv.set_index(["Ticker", "Date"]).sort_index()

    fed_path = GLOBAL_DATA_DIR / "High_Yield_Spread_T10Y2Y_Spread.csv"
    if fed_path.exists():
        df_fed = pd.read_csv(fed_path)
        if "Unnamed: 0" in df_fed.columns:
            df_fed = df_fed.rename(columns={"Unnamed: 0": "Date"})
        # ROOT LEVEL FIX: Clean Fed dates
        df_fed["Date"] = (
            pd.to_datetime(df_fed["Date"], utc=True).dt.tz_convert(None).dt.normalize()
        )
        df_fed = df_fed.drop_duplicates(subset=["Date"], keep="last")

        df_fed[["High_Yield_Spread", "Yield_Curve_10Y2Y"]] = df_fed[
            ["High_Yield_Spread", "Yield_Curve_10Y2Y"]
        ].astype("float32")

    indices_path = GLOBAL_DATA_DIR / "VIX3M_VIX.parquet"
    if indices_path.exists():
        df_indices = pd.read_parquet(indices_path).reset_index()
        # ROOT LEVEL FIX: Clean Indices dates
        df_indices["Date"] = (
            pd.to_datetime(df_indices["Date"], utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )
        df_indices = df_indices.drop_duplicates(subset=["Ticker", "Date"], keep="last")
        df_indices = df_indices.set_index(["Ticker", "Date"]).sort_index()

        df_indices[["Adj Open", "Adj High", "Adj Low", "Adj Close"]] = df_indices[
            ["Adj Open", "Adj High", "Adj Low", "Adj Close"]
        ].astype("float32")
        df_indices["Volume"] = df_indices["Volume"].astype("float32")

    return df_ohlcv, df_fed, df_indices


def load_processed_data() -> ProcessedDataBundle:
    """Loads aligned and preprocessed data needed for Cache building and Training."""
    return ProcessedDataBundle(
        df_ohlcv=pd.read_parquet(LOCAL_DATA_DIR / "df_ohlcv.parquet"),
        macro_df=pd.read_parquet(LOCAL_DATA_DIR / "macro_df.parquet"),
        features_df=pd.read_parquet(LOCAL_DATA_DIR / "features_df.parquet"),
    )
