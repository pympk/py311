import pytest
import pandas as pd


# Mocking the configuration to control the test environment
class MockThresholds:
    quality_min_periods = 2
    min_median_dollar_volume = 1_000_000


class MockConfig:
    thresholds = MockThresholds()
    calendar_ticker = "SPY"
    handle_zeros_as_nan = True
    max_data_gap_ffill = 1


@pytest.fixture
def dummy_ohlcv():
    """
    Creates a dummy MultiIndex DataFrame with 3 tickers:
    - AAPL: Liquid, trades 1996 to 1998 (Tests 1997 date truncation).
    - JUNK: Illiquid, never hits $1M volume (Tests pre-screening).
    - GHOST: Liquid, trades briefly in mid-1997 (Tests 'came and went' lifespan logic).
    """
    dates = pd.date_range(start="1996-12-30", end="1998-01-05", freq="B")

    data = []
    for d in dates:
        # AAPL: Always high volume
        data.append(
            {
                "Ticker": "AAPL",
                "Date": d,
                "Adj Close": 10.0,
                "Volume": 200000,
                "Adj Open": 10,
                "Adj High": 11,
                "Adj Low": 9,
            }
        )
        # JUNK: Low volume (Dollar Volume = $5,000)
        data.append(
            {
                "Ticker": "JUNK",
                "Date": d,
                "Adj Close": 5.0,
                "Volume": 1000,
                "Adj Open": 5,
                "Adj High": 5,
                "Adj Low": 5,
            }
        )

    # GHOST: Only trades for 3 days in July 1997 (High volume)
    ghost_dates = pd.date_range(start="1997-07-01", end="1997-07-03", freq="B")
    for d in ghost_dates:
        data.append(
            {
                "Ticker": "GHOST",
                "Date": d,
                "Adj Close": 20.0,
                "Volume": 100000,
                "Adj Open": 20,
                "Adj High": 21,
                "Adj Low": 19,
            }
        )

    # Inject a price gap in AAPL to test ffill
    # Remove AAPL's row for 1997-01-03
    data = [
        row
        for row in data
        if not (row["Ticker"] == "AAPL" and row["Date"] == pd.Timestamp("1997-01-03"))
    ]

    df = pd.DataFrame(data).set_index(["Ticker", "Date"])
    return df, dates


def test_cell3_logic(dummy_ohlcv):
    df_ohlcv, master_dates = dummy_ohlcv
    config = MockConfig()

    # --- 1. DATE TRUNCATION ---
    CUTOFF_DATE = pd.Timestamp("1997-01-01")
    df_ohlcv = df_ohlcv[df_ohlcv.index.get_level_values("Date") >= CUTOFF_DATE].copy()
    master_dates = master_dates[master_dates >= CUTOFF_DATE]

    # Assert Pre-1997 data is dropped
    assert (
        df_ohlcv.index.get_level_values("Date").min() >= CUTOFF_DATE
    ), "Data before 1997 was not dropped"

    # --- 2. FAST PRE-SCREENING ---
    df_ohlcv["DollarVolume"] = df_ohlcv["Adj Close"] * df_ohlcv["Volume"]
    max_med_dv = (
        df_ohlcv.groupby(level="Ticker")["DollarVolume"]
        .rolling(window=config.thresholds.quality_min_periods, min_periods=1)
        .median()
        .groupby(level=0)
        .max()
    )

    valid_tickers = max_med_dv[
        max_med_dv >= config.thresholds.min_median_dollar_volume
    ].index
    df_ohlcv = df_ohlcv[
        df_ohlcv.index.get_level_values("Ticker").isin(valid_tickers)
    ].copy()
    df_ohlcv.drop(columns=["DollarVolume"], inplace=True)

    # Assert JUNK is dropped, AAPL and GHOST remain
    remaining_tickers = df_ohlcv.index.get_level_values("Ticker").unique()
    assert "JUNK" not in remaining_tickers, "Illiquid stock was not filtered out"
    assert "AAPL" in remaining_tickers, "Valid stock was incorrectly filtered out"
    assert "GHOST" in remaining_tickers, "Short-lifespan valid stock was filtered out"

    # --- 3. SMART FORWARD FILL ---
    def fill_ticker(df_ticker):
        min_d = df_ticker.index.get_level_values("Date").min()
        max_d = df_ticker.index.get_level_values("Date").max()
        stock_calendar = master_dates[(master_dates >= min_d) & (master_dates <= max_d)]

        df_ticker = df_ticker.droplevel("Ticker").reindex(stock_calendar).copy()
        price_cols = ["Adj Open", "Adj High", "Adj Low", "Adj Close"]

        df_ticker.loc[:, price_cols] = df_ticker[price_cols].ffill(
            limit=config.max_data_gap_ffill
        )  # <--- Using .loc[:, ...]

        df_ticker["Volume"] = df_ticker["Volume"].fillna(0)
        return df_ticker

    df_ohlcv = df_ohlcv.groupby(level="Ticker").apply(fill_ticker)
    if isinstance(df_ohlcv.index, pd.MultiIndex):
        df_ohlcv.index.names = ["Ticker", "Date"]

    # Assert AAPL's missing date (1997-01-03) was forward filled
    aapl_data = df_ohlcv.xs("AAPL", level="Ticker")
    assert (
        pd.Timestamp("1997-01-03") in aapl_data.index
    ), "Missing date was not injected via reindex"
    assert not pd.isna(
        aapl_data.loc["1997-01-03", "Adj Close"]
    ), "Price was not forward filled"
    assert aapl_data.loc["1997-01-03", "Volume"] == 0, "Volume for gap day should be 0"

    # Assert GHOST does NOT have dates outside its July 1st-3rd lifespan
    ghost_data = df_ohlcv.xs("GHOST", level="Ticker")
    assert ghost_data.index.min() == pd.Timestamp(
        "1997-07-01"
    ), "GHOST min date was expanded incorrectly"
    assert ghost_data.index.max() == pd.Timestamp(
        "1997-07-03"
    ), "GHOST max date was expanded incorrectly"
    assert (
        len(ghost_data) == 3
    ), "GHOST has rows outside of its lifespan (Cartesian bloat)"
