import os
import regutil
import time
import datetime
import numpy as np
import pandas as pd
import empyrical
import warnings
import re
from IPython.display import display, Markdown


warnings.filterwarnings("ignore", message='Module "zipline.assets" not found.*')


def get_latest_downloaded_files(directory_path, num_files=10):
    """
    Returns the N most recent files in a directory, sorted by modification time.

    Args:
        directory_path (str): The path to the directory to search.
        num_files (int): The number of files to list (default: 10).

    Returns:
        list: A list of tuples, where each tuple contains:
              (filename, file_size_bytes, last_modified_time)
              Returns an empty list if the directory doesn't exist or is empty.
    """

    # Check if the directory exists
    if not os.path.exists(directory_path):
        return []

    # Get a list of all files in the directory
    file_list = []
    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            # Check if the file is a regular file (not a directory)
            if os.path.isfile(filepath):
                # Get the size of the file
                file_size = os.path.getsize(filepath)
                # Get the last modification time of the file
                last_modified_time = os.path.getmtime(filepath)
                # Add the file to the list
                file_list.append((filename, file_size, last_modified_time))

        # Sort the list of files by modification time
        file_list.sort(key=lambda x: x[2], reverse=True)

        # Return the top N files
        return file_list[:num_files]
    except OSError:
        # If there is an error, return an empty list
        return []


def calculate_performance_metrics(returns, risk_free_rate=0.0):
    """
    Calculates Sortino Ratio, Sharpe Ratio, and Omega Ratio using PyFolio/Empyrical.

    Args:
        returns (pd.Series or np.array):  Daily returns of the investment.
                                         Must be a Pandas Series with a DatetimeIndex.
        risk_free_rate (float):  The risk-free rate (annualized). Default is 0.0.

    Returns:
        dict: A dictionary containing the calculated ratios with modified keys.
              Returns None if there is an error or the input is invalid.
    """

    try:
        # Ensure returns is a pandas Series with a DatetimeIndex.  Crucial for pyfolio.
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)  # Convert to Series if needed
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must be a Pandas Series with a DatetimeIndex.")

        # Convert annualized risk-free rate to daily rate
        days_per_year = 252  # Standard for financial calculations
        daily_risk_free_rate = risk_free_rate / days_per_year

        # Temporarily suppress the warning just for this calculation
        # if returns are the same, std=0, sharpe ratio is either inf or -inf
        # if (returns - daily_risk_free_rate) are all positive, sortino and omega ratios are infinite
        with np.errstate(divide="ignore", invalid="ignore"):
            # Calculate the Sharpe Ratio using empyrical (as pyfolio's is deprecated)
            sharpe_ratio = empyrical.sharpe_ratio(
                returns, risk_free=daily_risk_free_rate, annualization=days_per_year
            )

            # Calculate the Sortino Ratio using empyrical
            sortino_ratio = empyrical.sortino_ratio(
                returns,
                required_return=daily_risk_free_rate,
                annualization=days_per_year,
            )

            # Calculate the Omega Ratio using empyrical
            omega_ratio = empyrical.omega_ratio(
                returns, risk_free=daily_risk_free_rate, annualization=days_per_year
            )

        n = len(returns) + 1  # Calculate n only once

        return {
            f"Sharpe {n}d": sharpe_ratio,
            f"Sortino {n}d": sortino_ratio,
            f"Omega {n}d": omega_ratio,
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def calculate_returns(adj_close_prices):
    """
    Calculates daily returns from adjusted close prices.

    Args:
        adj_close_prices (pd.Series): Pandas Series of adjusted close prices with DatetimeIndex.

    Returns:
        pd.Series: Pandas Series of daily returns with DatetimeIndex, sorted by date (oldest to newest).
    """
    try:
        if not isinstance(adj_close_prices, pd.Series):
            raise TypeError("Input must be a Pandas Series.")
        if not isinstance(adj_close_prices.index, pd.DatetimeIndex):
            raise ValueError("Input Series must have a DatetimeIndex.")

        # Sort the index to ensure correct return calculation (oldest to newest)
        adj_close_prices = adj_close_prices.sort_index()

        # Calculate daily returns using pct_change()
        returns = adj_close_prices.pct_change().dropna()  # Drop the first NaN value

        return returns

    except Exception as e:
        print(f"Error calculating returns: {e}")
        return None


def analyze_stock(df, ticker, risk_free_rate=0.0, output_debug_data=False):
    """
    Analyzes a single stock's performance based on its adjusted close prices.

    Args:
        df (pd.DataFrame): MultiIndex DataFrame. df.index.levels[0] are symbols,
        df_cleaned_OHLCV.index.levels[1] are dates.
        df containing stock's Open, High, Low, Close, Adj Close, Volume, Adj Open, Adj High, Adj Low.
        ticker (str): The stock ticker symbol (e.g., 'NVDA').
        risk_free_rate (float): The annualized risk-free rate. Default is 0.0.
        output_debug_data (bool): If True, print Adj Close prices and returns (default: False).

    Returns:
        pd.DataFrame: A DataFrame with the ticker as index and 'Sharpe Ratio', 'Sortino Ratio', and 'Omega Ratio' as columns.
                       Returns None if there is an error.  Crucially changed to return None, not an empty dataframe.
    """
    try:
        # Extract Adj Close prices for ticker, sorted by date oldest to newest
        adj_close_prices = df.loc[ticker]["Adj Close"].sort_index()

        # Check if adj_close_prices is a Series
        if not isinstance(adj_close_prices, pd.Series):
            raise TypeError(
                f"Expected a Pandas Series for Adj Close prices of {ticker}. Check that {ticker} exists in the DataFrame, and that 'Adj Close' is a valid column"
            )

        # Calculate returns
        returns_series = calculate_returns(adj_close_prices)

        if returns_series is not None:
            # Output debug data if requested
            if output_debug_data:
                print(f"--- Debug Data for {ticker} ---")
                print("\nAdj Close Prices (Dates and Values):")
                print(
                    adj_close_prices
                )  # This is a Series, prints the index(dates) and values.
                print("\nReturns:")
                print(returns_series)

            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics(
                returns_series, risk_free_rate=risk_free_rate
            )

            if performance_metrics:
                # Create a DataFrame from the metrics
                metrics_df = pd.DataFrame(performance_metrics, index=[ticker])
                return metrics_df
            else:
                print(f"Could not calculate performance metrics for {ticker}.")
                return None  # Return None, not an empty DataFrame
        else:
            print(f"Failed to calculate returns for {ticker}.")
            return None  # Return None, not an empty DataFrame

    except KeyError:
        print(f"Ticker '{ticker}' not found in DataFrame.")
        return None  # Return None, not an empty DataFrame
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None  # Return None, not an empty DataFrame


def get_recent_rows(df, num_rows=None):
    """
    Returns a DataFrame containing the most recent specified number of rows for each symbol in the input DataFrame.
    If num_rows is None, all rows for each symbol are returned.

    Args:
        df: A pandas DataFrame with a MultiIndex, where the first level is the symbol and the second level is a Timestamp representing the date.
        num_rows: The number of recent rows to include for each symbol.
                  If None (default), all rows for each symbol are returned.

    Returns:
        A pandas DataFrame containing the most recent specified number of rows for each symbol.
        Returns an empty DataFrame if df is empty.
    """

    if df.empty:
        return pd.DataFrame()

    # Sort the DataFrame by symbol and date (within each symbol)
    df = df.sort_index()

    if num_rows is None:
        return df  # Return the entire DataFrame if num_rows is None

    def get_last_n_rows(group):
        return group.tail(num_rows)

    # Group by symbol and apply the function to get the last n rows for each group
    recent_df = df.groupby(level=0, group_keys=False).apply(get_last_n_rows)

    return recent_df


def find_sorted_intersection(list1, list2):
    """
    Finds the intersection of two lists and returns a new sorted list containing
    the common elements.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        list: A new sorted list containing the common elements of list1 and list2.
              Returns an empty list if there are no common elements or if either
              input list is None or empty.
    """

    if not list1 or not list2:  # Check for empty or None lists
        return []

    # Convert lists to sets for efficient intersection finding
    set1 = set(list1)
    set2 = set(list2)

    # Find the intersection using set intersection
    intersection_set = set1.intersection(set2)

    # Convert the intersection set to a list and sort it
    intersection_list = sorted(list(intersection_set))

    return intersection_list


def filter_df_dates_to_reference_symbol(df, reference_symbol="AAPL"):
    """
    Filters symbols in a DataFrame based on date index matching a reference symbol (default AAPL)
    and provides analysis of the filtering results.

    Args:
        df (pd.DataFrame): DataFrame with a MultiIndex ('Symbol', 'Date').
        reference_symbol (str): The symbol to use as the reference for date comparison. Defaults to "AAPL".

    Returns:
        pd.DataFrame: The filtered DataFrame.  Prints analysis to standard output.
    """

    # Get the date index for the reference symbol.  Return empty DataFrame if symbol not found
    try:
        reference_dates = df.loc[reference_symbol].index
    except KeyError:
        print(f"Error: Reference symbol '{reference_symbol}' not found in DataFrame.")
        return (
            pd.DataFrame()
        )  # Return an empty DataFrame if reference_symbol is not found

    original_symbols = df.index.get_level_values("Symbol").unique().tolist()

    # Filter symbols based on date index matching with the reference symbol
    filtered_symbols = []
    for symbol in original_symbols:
        try:  # Handle the case where a symbol might be missing from the df
            symbol_dates = df.loc[symbol].index
        except KeyError:
            continue  # Skip to the next symbol if this one is missing

        if len(symbol_dates) == len(reference_dates) and symbol_dates.equals(
            reference_dates
        ):
            filtered_symbols.append(symbol)

    # Create the filtered DataFrame
    df_filtered = df.loc[filtered_symbols]

    # Analyze the filtering results
    print(f"Original number of symbols: {len(original_symbols)}")
    print(f"Number of symbols after filtering: {len(filtered_symbols)}")
    print(
        f"Number of symbols filtered out: {len(original_symbols) - len(filtered_symbols)}"
    )

    filtered_out_symbols = list(set(original_symbols) - set(filtered_symbols))

    print("\nFirst 10 symbols that were filtered out:")
    print(filtered_out_symbols[:10])

    if filtered_out_symbols:
        print("\nExample of dates for first filtered out symbol:")
        first_filtered_symbol = filtered_out_symbols[0]
        try:  # Handle potential KeyError if the symbol doesn't exist (e.g., due to filtering earlier)
            print(f"\nDates for {first_filtered_symbol}:")
            print(df.loc[first_filtered_symbol].index)
        except KeyError:
            print(
                f"\nSymbol '{first_filtered_symbol}' not found in the original DataFrame."
            )

    print("\nFiltered DataFrame info:")
    print(df_filtered.info())

    return df_filtered


def get_latest_dfs(df, num_rows):
    """
    Get the latest N rows for each symbol from a multi-index DataFrame,
    returning multiple filtered DataFrames in a list.

    Processes a DataFrame with multi-level index (Symbol, Date) to return:
    - For each number in num_rows: A DataFrame containing the latest N dates
      for each symbol, sorted by symbol and chronological date order

    Parameters:
    df (pd.DataFrame): Input DataFrame with MultiIndex of (Symbol, Date)
    num_rows (list[int]): List of integers specifying numbers of recent rows to return

    Returns:
    list[pd.DataFrame]: List of filtered DataFrames in the same order as num_rows
                        Example: [df_30, df_60] for num_rows=[30, 60]

    Raises:
    KeyError: If DataFrame doesn't have 'Symbol' and 'Date' index levels
    TypeError: If num_rows contains non-integer values

    Example:
    >>> df = pd.DataFrame(index=pd.MultiIndex.from_product(
    ...     [['AAPL', 'MSFT'], pd.date_range('2020-01-01', periods=100)],
    ...     names=['Symbol', 'Date']
    ... ))
    >>> dfs = get_latest_dfs(df, [30, 60])
    >>> len(dfs[0].loc['AAPL'])
    30
    >>> len(dfs[1].loc['MSFT'])
    60
    """
    result_list = []

    # Validate input types
    if not all(isinstance(n, int) for n in num_rows):
        raise TypeError("All elements in num_rows must be integers")

    for num in num_rows:
        # Group by symbol and process each group
        result = df.groupby(level="Symbol", group_keys=False).apply(
            lambda group:
            # Sort group dates descending and take top N rows
            group.sort_index(level="Date", ascending=False).head(num)
        )

        # Global sort for final output:
        # 1. Symbols in alphabetical order (ascending=True)
        # 2. Dates in chronological order (ascending=True) within each symbol
        result = result.sort_index(level=["Symbol", "Date"], ascending=[True, True])

        result_list.append(result)

    return result_list


def filter_symbols_with_missing_values(df):
    """
    Filters out symbols from a MultiIndex DataFrame that have:
    1. Any missing values in any columns
    2. Missing any dates present in the original DataFrame's date index

    Args:
        df (pd.DataFrame): A pandas DataFrame with a MultiIndex, where the first
                           level is the symbol and the second level is the date.

    Returns:
        tuple: A tuple containing:
            - filtered_df (pd.DataFrame): A DataFrame containing only the symbols
              without missing values or dates. Returns empty if none are clean.
            - symbols_with_missing (list): List of symbols with missing data/dates.
    """
    if df.empty:
        return pd.DataFrame(), []

    # Get all unique dates from the original DataFrame
    expected_dates = df.index.levels[1].unique()

    symbols_with_missing = []
    filtered_data = []
    kept_symbols = []

    for symbol in df.index.get_level_values(0).unique():
        symbol_df = df.loc[symbol]

        # Check for missing values
        has_nan = symbol_df.isnull().any().any()

        # Check for missing dates
        missing_dates = expected_dates.difference(symbol_df.index)

        if has_nan or len(missing_dates) > 0:
            symbols_with_missing.append(symbol)
        else:
            filtered_data.append(symbol_df)
            kept_symbols.append(symbol)

    if filtered_data:
        filtered_df = pd.concat(
            filtered_data, keys=kept_symbols, names=["Symbol", "Date"]
        )
    else:
        filtered_df = pd.DataFrame()

    return filtered_df, symbols_with_missing


def convert_volume(value):
    """
    Convert a string volume representation with suffix (M/K) to float millions.

    Handles values like '51.73M' (51.73 million) or '111.53K' (0.11153 million),
    converting them to float values in millions unit.

    Parameters:
    value (str): Input string value to convert. Can contain 'M' (million)
                 or 'K' (thousand) suffix. Also handles NaN values.

    Returns:
    float: Numerical value in millions. Returns np.nan for invalid formats,
           unknown suffixes, or non-string inputs.

    Example:
    >>> convert_volume('51.73M')
    51.73
    >>> convert_volume('111.53K')
    0.11153
    >>> convert_volume('invalid')
    nan
    """
    # Handle missing values immediately
    if pd.isna(value):
        return np.nan

    try:
        # Clean and standardize input: remove whitespace, make uppercase
        cleaned = value.strip().upper()

        # Extract suffix (last character) and numerical part
        suffix = cleaned[-1]
        number_str = cleaned[:-1]

        # Convert numerical part to float
        number = float(number_str)

        # Apply conversion based on suffix
        if suffix == "M":
            # Already in millions, return directly
            return number
        elif suffix == "K":
            # Convert thousands to millions (divide by 1000)
            return number / 1000
        else:
            # Return NaN for unknown suffixes (e.g., 'B', 'T')
            return np.nan

    except (ValueError, IndexError, TypeError):
        # Handle various error scenarios:
        # - ValueError: if number conversion fails
        # - IndexError: if string is empty after cleaning
        # - TypeError: if input isn't string-like
        return np.nan


def extract_date_from_string(
    input_string, pattern=r"(\d{4}-\d{2}-\d{2})", group_index=1
):
    """
    Extracts a date (or any matching pattern) from a string using a regular expression.

    Args:
        input_string (str): The string to search within.
        pattern (str): The regular expression pattern to use for extraction.  Defaults to YYYY-MM-DD.
        group_index (int): The index of the capturing group in the regex pattern that contains the desired date.  Defaults to 1.

    Returns:
        str: The extracted date string if found, otherwise raises a ValueError.

    Raises:
        ValueError: If no match is found for the specified pattern in the input string.
    """

    match = re.search(pattern, input_string)

    if not match:
        raise ValueError(
            f"No match found for pattern '{pattern}' in input string: '{input_string}'"
        )

    try:
        extracted_value = match.group(group_index)
        return extracted_value
    except IndexError:
        raise ValueError(
            f"Invalid group index: {group_index}. The pattern '{pattern}' does not have this many capturing groups."
        )


def get_matching_files(dir, create_dir=True, start_file_pattern="df_OHLCV_"):
    """Return list of files matching specified pattern in directory"""
    if create_dir:
        os.makedirs(dir, exist_ok=True)
    try:
        return [f for f in os.listdir(dir) if f.startswith(start_file_pattern)]
    except FileNotFoundError:
        return []


def process_downloads_dir(downloads_dir, limit=20, start_file_pattern="df_OHLCV_"):
    """Process Downloads directory with pattern-based filtering"""
    try:
        all_files = []
        for f in os.listdir(downloads_dir):
            file_path = os.path.join(downloads_dir, f)
            if os.path.isfile(file_path):
                all_files.append(file_path)

        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_files = all_files[:limit]
        matched_files = [
            f
            for f in latest_files
            if os.path.basename(f).startswith(start_file_pattern)
        ]

        msg = (
            f"<span style='color:#00ffff;font-weight:500'>[Downloads] Scanned latest {len(latest_files)} files • "
            f"Found {len(matched_files)} '{start_file_pattern}' matches</span>"
        )
        display(Markdown(msg))
        return matched_files

    except Exception as e:
        display(
            Markdown(
                f"<span style='color:red'>Error accessing Downloads: {str(e)}</span>"
            )
        )
        return []


def display_file_selector(files_with_source, start_file_pattern):
    """Show interactive file selector with dynamic pattern"""
    display(Markdown(f"**Available '{start_file_pattern}' files:**"))

    for idx, (file_path, source) in enumerate(files_with_source, 1):
        name = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        timestamp = os.path.getmtime(file_path)

        size_mb = size / (1024 * 1024)
        formatted_date = datetime.datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M"
        )

        file_info = (
            f"- ({idx}) `[{source.upper()}]` `{name}` "
            f"<span style='color:#00ffff'>"
            f"({size_mb:.2f} MB, {formatted_date})"
            f"</span>"
        )
        display(Markdown(file_info))

    print(f"\nInput a number to select file (1-{len(files_with_source)})")


def generate_clean_filename(source_file):
    """Create cleaned filename with _clean suffix"""
    base, ext = os.path.splitext(source_file)
    return f"{base}_clean{ext}"


def get_user_choice(files_with_source):
    """Handle user input with validation"""
    while True:
        try:
            prompt = f"Select file (1-{len(files_with_source)}):"
            choice = int(input(prompt))
            if 1 <= choice <= len(files_with_source):
                return files_with_source[choice - 1][0]  # Return the path from tuple
            display(
                Markdown(
                    f"<span style='color:red'>Enter 1-{len(files_with_source)}</span>"
                )
            )
        except ValueError:
            display(Markdown("<span style='color:red'>Numbers only!</span>"))


def get_cov_corr_ewm_matrices(df, span=21, return_corr=True, return_cov=True):
    """
    Calculates the exponentially weighted moving (EWM) covariance and/or correlation matrix,
    correcting for potential biases introduced by standard EWM calculations
    and handling cases with zero variance.  Can return either or both matrices.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.  Each column represents a different asset.
        span (int, optional): The span parameter for the EWM calculation.  A larger span
                             gives more weight to recent data. Defaults to 21.
        return_corr (bool, optional): Whether to return the correlation matrix. Defaults to True.
        return_cov (bool, optional): Whether to return the covariance matrix. Defaults to True.

    Returns:
        tuple: A tuple containing the EWM covariance matrix and/or the correlation matrix,
               depending on the values of `return_cov` and `return_corr`.  Returns None for a matrix if the
               corresponding `return_` flag is False. The order of the matrices in the tuple is
               (covariance_matrix, correlation_matrix).

               If both `return_corr` and `return_cov` are False, returns None.
    """
    alpha = 2 / (span + 1)

    ewm_mean = df.ewm(alpha=alpha, adjust=False).mean()
    demeaned = df - ewm_mean

    # Compute weights for valid observations
    weights = (1 - alpha) ** np.arange(len(df), 0, -1)
    weights /= weights.sum()

    # Compute covariance using clean data
    cov_matrix = np.einsum(
        "t,tij->ij", weights, np.einsum("ti,tj->tij", demeaned.values, demeaned.values)
    )

    if return_corr:
        # Handle zero variances to avoid division by zero
        variances = np.diag(cov_matrix).copy()
        variances[variances <= 0] = 1e-10  # Prevent NaN/Inf during normalization
        std_devs = np.sqrt(variances)

        # Calculate correlation matrix
        correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
        correlation_matrix = pd.DataFrame(
            correlation_matrix, index=df.columns, columns=df.columns
        )
    else:
        correlation_matrix = None

    cov_matrix = (
        pd.DataFrame(cov_matrix, index=df.columns, columns=df.columns)
        if return_cov
        else None
    )

    if not return_corr and not return_cov:
        return None

    return (
        (cov_matrix, correlation_matrix)
        if return_cov and return_corr
        else (cov_matrix if return_cov else correlation_matrix)
    )


def get_cov_corr_ewm_matrices_chunked(
    df, span=21, return_corr=True, return_cov=True, chunk_size=100
):
    """
    Robust chunked calculation of EWM covariance and correlation matrices.
    Handles edge cases and ensures proper broadcasting.
    """
    alpha = 2 / (span + 1)

    # Clean data - remove inf and drop rows with any NaN
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n_assets = len(clean_df.columns)
    n_obs = len(clean_df)

    # Calculate EWM mean and demean
    ewm_mean = clean_df.ewm(alpha=alpha, adjust=False).mean()
    demeaned = clean_df - ewm_mean

    # Compute weights as a column vector
    weights = (1 - alpha) ** np.arange(n_obs, 0, -1)
    weights /= weights.sum()
    weights = weights.reshape(-1, 1)  # Shape (n_obs, 1)

    # Initialize covariance matrix
    cov_matrix = np.zeros((n_assets, n_assets))

    # Process in chunks
    for i in range(0, n_assets, chunk_size):
        i_end = min(i + chunk_size, n_assets)
        chunk_i = demeaned.iloc[:, i:i_end].values  # Shape (n_obs, chunk_size)

        # Apply weights to chunk_i (broadcasting works automatically)
        weighted_chunk_i = chunk_i * weights  # Shape (n_obs, chunk_size)

        for j in range(i, n_assets, chunk_size):  # Start from i for upper triangle
            j_end = min(j + chunk_size, n_assets)
            chunk_j = demeaned.iloc[:, j:j_end].values  # Shape (n_obs, chunk_size)

            # Calculate weighted products for this chunk pair
            cov_chunk = np.dot(
                weighted_chunk_i.T, chunk_j
            )  # Shape (chunk_size, chunk_size)

            # Fill the covariance matrix
            cov_matrix[i:i_end, j:j_end] = cov_chunk

            # Fill symmetric part if not on diagonal
            if i != j:
                cov_matrix[j:j_end, i:i_end] = cov_chunk.T

    # Prepare results
    results = []

    if return_cov:
        cov_matrix_df = pd.DataFrame(
            cov_matrix, index=clean_df.columns, columns=clean_df.columns
        )
        results.append(cov_matrix_df)

    if return_corr:
        # Handle zero variances
        variances = np.diag(cov_matrix).copy()
        variances[variances <= 0] = 1e-10  # Small positive value
        std_devs = np.sqrt(variances)

        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        corr_matrix_df = pd.DataFrame(
            corr_matrix, index=clean_df.columns, columns=clean_df.columns
        )
        results.append(corr_matrix_df)

    return tuple(results) if len(results) > 1 else results[0]


# def main_processor_(data_dir='../data', downloads_dir=None, downloads_limit=20,
#                    clean_name_override=None, start_file_pattern='df_OHLCV_',
#                    contains_pattern=None):
#     """
#     Orchestrate file selection with configurable start and contains patterns.
#     Uses the base name of data_dir as the origin label for its files.

#     Args:
#         data_dir (str): Path to the primary data directory.
#         downloads_dir (str, optional): Path to the downloads directory. Defaults to ~/Downloads.
#         downloads_limit (int): Maximum number of files to consider from downloads.
#         clean_name_override (str, optional): If provided, overrides the generated destination filename.
#         start_file_pattern (str): Pattern for the start of filenames to search for.
#         contains_pattern (str, optional): Additional pattern that must be contained within the filename.
#                                          Defaults to None (no contains filtering).

#     Returns:
#         tuple: (selected_file_path, destination_path) or (None, None) if no files found or selected.
#     """
#     if downloads_dir is None:
#         downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')

#     # Normalize data_dir path for consistent processing and get its base name
#     data_dir_abs = os.path.abspath(data_dir)
#     data_dir_label = os.path.basename(data_dir_abs) or data_dir_abs # Use basename, fallback to full path if basename is empty (e.g., root dir)

#     # Get data directory files matching BOTH patterns
#     data_files_raw = get_matching_files(data_dir_abs, create_dir=True, start_file_pattern=start_file_pattern)
#     data_files = [(os.path.join(data_dir_abs, f), data_dir_label) # <-- Use base name as origin label
#                   for f in data_files_raw
#                   if not contains_pattern or contains_pattern in f] # Apply contains_pattern filter

#     # Get downloads files matching BOTH patterns
#     downloads_files = []
#     if os.path.exists(downloads_dir):
#         raw_downloads = process_downloads_dir(downloads_dir, downloads_limit, start_file_pattern=start_file_pattern)
#         # Apply contains_pattern filter to downloads files (checking basename)
#         filtered_downloads = [f for f in raw_downloads
#                               if not contains_pattern or contains_pattern in os.path.basename(f)]
#         downloads_files = [(f, 'downloads') for f in filtered_downloads] # Keep 'downloads' label distinct

#     ohlcv_files = data_files + downloads_files

#     # Construct informative error/selection messages
#     search_criteria = f"'{start_file_pattern}'"
#     if contains_pattern:
#         search_criteria += f" and containing '{contains_pattern}'"

#     if not ohlcv_files:
#         display(Markdown(f"**Error:** No files found matching {search_criteria}! "
#                          f"(Searched: '{data_dir_label}' dir and 'downloads')")) # <-- Use label in error msg
#         return None, None

#     # Pass the combined search criteria description to display
#     display_file_selector(ohlcv_files, search_criteria)
#     selected_file = get_user_choice(ohlcv_files)

#     if selected_file is None: # Handle case where user cancels
#          display(Markdown(f"**Info:** No file selected."))
#          return None, None

#     clean_name = generate_clean_filename(os.path.basename(selected_file))
#     if clean_name_override is not None:
#         clean_name = clean_name_override
#     # Destination path still uses the full absolute path for reliability
#     dest_path = os.path.join(data_dir_abs, clean_name)

#     display(Markdown(f"""
#     **Selected paths:**
#     - Source: `{selected_file}`
#     - Destination: `{dest_path}`
#     """))
#     return selected_file, dest_path


def main_processor(
    data_dir="../data",
    downloads_dir=None,
    downloads_limit=20,
    clean_name_override=None,
    start_file_pattern="df_OHLCV_",
    contains_pattern=None,
):
    """
    Orchestrate file selection with configurable start and contains patterns.
    Uses the base name of data_dir as the origin label for its files. Also
    returns the list of basenames of files presented to the user.

    Args:
        data_dir (str): Path to the primary data directory.
        downloads_dir (str, optional): Path to the downloads directory. Defaults to ~/Downloads.
                                      Set to '' or None to skip searching downloads.
        downloads_limit (int): Maximum number of files to consider from downloads.
        clean_name_override (str, optional): If provided, overrides the generated destination filename.
        start_file_pattern (str): Pattern for the start of filenames to search for.
        contains_pattern (str, optional): Additional pattern that must be contained within the filename.
                                         Defaults to None (no contains filtering).

    Returns:
        tuple: (selected_file_path, destination_path, displayed_filenames_list)
               where displayed_filenames_list is a list of strings (basenames only).
               Returns (None, None, []) if no files are found.
               Returns (None, None, displayed_filenames_list) if the user cancels selection.
    """
    if downloads_dir is None:
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    elif downloads_dir == "":  # Treat empty string as explicitly skipping downloads
        downloads_dir = None

    # Normalize data_dir path for consistent processing and get its base name
    data_dir_abs = os.path.abspath(data_dir)
    data_dir_label = (
        os.path.basename(data_dir_abs) or data_dir_abs
    )  # Use basename, fallback to full path if basename is empty (e.g., root dir)

    # Get data directory files matching BOTH patterns
    data_files_raw = get_matching_files(
        data_dir_abs, create_dir=True, start_file_pattern=start_file_pattern
    )
    data_files = [
        (
            os.path.join(data_dir_abs, f),
            data_dir_label,
        )  # <-- Use base name as origin label
        for f in data_files_raw
        if not contains_pattern or contains_pattern in f
    ]  # Apply contains_pattern filter

    # Get downloads files matching BOTH patterns
    downloads_files = []
    # Only search downloads if the path exists and was provided (not None or '')
    if downloads_dir and os.path.exists(downloads_dir):
        raw_downloads = process_downloads_dir(
            downloads_dir, downloads_limit, start_file_pattern=start_file_pattern
        )
        # Apply contains_pattern filter to downloads files (checking basename)
        filtered_downloads = [
            f
            for f in raw_downloads
            if not contains_pattern or contains_pattern in os.path.basename(f)
        ]
        downloads_files = [
            (f, "downloads") for f in filtered_downloads
        ]  # Keep 'downloads' label distinct
    elif downloads_dir and not os.path.exists(downloads_dir):
        display(
            Markdown(
                f"**Warning:** Downloads directory specified but not found: `{downloads_dir}`"
            )
        )

    # Internal list still uses tuples for display logic
    ohlcv_files = data_files + downloads_files

    # Construct informative error/selection messages
    search_criteria = f"starting with '{start_file_pattern}'"
    if contains_pattern:
        search_criteria += f" and containing '{contains_pattern}'"

    # Create the list of basenames to be returned
    displayed_filenames = [os.path.basename(f_path) for f_path, _ in ohlcv_files]

    if not ohlcv_files:  # Check the original tuple list
        display(
            Markdown(
                f"**Error:** No files found matching {search_criteria}! "
                f"(Searched: '{data_dir_label}' dir and downloads (if applicable))"
            )
        )
        # Return None, None, and an empty list for displayed_filenames
        return None, None, []  # Return empty list [] directly

    # Pass the tuple list to display (it needs the origin info)
    display_file_selector(ohlcv_files, search_criteria)
    selected_file = get_user_choice(
        ohlcv_files
    )  # This returns the selected path or None

    if selected_file is None:  # Handle case where user cancels
        display(Markdown(f"**Info:** No file selected."))
        # Return None, None, but DO return the list of filenames that *were* displayed
        return None, None, displayed_filenames

    # --- File was selected ---
    clean_name = generate_clean_filename(os.path.basename(selected_file))
    if clean_name_override is not None:
        clean_name = clean_name_override
    # Destination path still uses the full absolute path for reliability
    dest_path = os.path.join(data_dir_abs, clean_name)

    display(
        Markdown(
            f"""
    **Selected paths:**
    - Source: `{selected_file}`
    - Destination: `{dest_path}`
    """
        )
    )

    # Return the selected file, destination path, AND the list of displayed *basenames*
    return selected_file, dest_path, displayed_filenames


#########################
from typing import Dict, Any


def print_stock_selection_report(output: Dict[str, Any]) -> None:
    """
    Prints a detailed report summarizing the results of the stock selection process,
    extracting all necessary information from the output dictionary.

    Args:
        output (Dict[str, Any]): The dictionary returned by the
                                select_stocks_from_clusters function, containing:
                                - 'selected_stocks': DataFrame of selected stocks.
                                - 'cluster_performance': DataFrame of selected cluster metrics.
                                - 'parameters': Dictionary of the input parameters used.
                                - 'cluster_stats_df': Original cluster stats DataFrame.
                                - 'detailed_clusters_df': Original detailed clusters DataFrame.
    Returns:
        None: This function prints output to the console.
    """
    # Extract data from the output dictionary using .get() for safety
    selected_stocks = output.get("selected_stocks", pd.DataFrame())
    cluster_performance = output.get("cluster_performance", pd.DataFrame())
    used_params = output.get("parameters", {})
    # Extract the input DataFrames needed for the report
    # cluster_stats_df = output.get('input_cluster_stats_df') # Might be None
    cluster_stats_df = output.get("cluster_stats_df")  # Might be None
    # detailed_clusters_df = output.get('input_detailed_clusters_df') # Might be None
    detailed_clusters_df = output.get("detailed_clusters_df")  # Might be None

    # --- Start of Original Code Block (adapted) ---

    print("\n=== CLUSTER SELECTION CRITERIA ===")
    print(
        "* Using Composite_Cluster_Score (balancing Raw Score and diversification) for cluster ranking."
    )
    print("* Using Risk_Adj_Score for stock selection within clusters.")

    num_selected_clusters = (
        len(cluster_performance) if not cluster_performance.empty else 0
    )
    # Use the extracted cluster_stats_df
    total_clusters = (
        len(cluster_stats_df)
        if cluster_stats_df is not None and not cluster_stats_df.empty
        else "N/A"
    )

    print(
        f"* Selected top {num_selected_clusters} clusters from {total_clusters} total initial clusters."
    )  # Adjusted wording slightly
    print(f"* Selection Criteria:")
    if used_params:
        for key, value in used_params.items():
            # Avoid printing the large input dataframes stored in parameters if they were added there too
            if not isinstance(value, pd.DataFrame):
                print(f"    {key}: {value}")
    else:
        print("    Parameters not available.")

    if not cluster_performance.empty:
        print("\n=== SELECTED CLUSTERS (RANKED BY COMPOSITE SCORE) ===")
        display_cols_exist = [
            col
            for col in [
                "Cluster_ID",
                "Size",
                "Avg_Raw_Score",
                "Avg_Risk_Adj_Score",
                "Avg_IntraCluster_Corr",
                "Avg_Volatility",
                "Composite_Cluster_Score",
                "Stocks_Selected",
                "Intra_Cluster_Diversification",
            ]
            if col in cluster_performance.columns
        ]
        print(
            cluster_performance[display_cols_exist]
            .sort_values("Composite_Cluster_Score", ascending=False)
            .to_string(index=False)
        )

        # Print top 8 stocks by Raw_Score for each selected cluster
        # Check if detailed_clusters_df was successfully extracted
        if detailed_clusters_df is not None and not detailed_clusters_df.empty:
            print("\n=== TOP STOCKS BY RAW SCORE PER SELECTED CLUSTER ===")
            print(
                """* Volatility is the standard deviation of daily returns over the past 250 trading days (example context).
* Note: The stocks below are shown ranked by Raw_Score for analysis,
*       but actual selection within the cluster was based on Risk_Adj_Score."""
            )

            for cluster_id in cluster_performance["Cluster_ID"]:
                cluster_stocks = detailed_clusters_df[
                    detailed_clusters_df["Cluster_ID"] == cluster_id
                ]
                if not cluster_stocks.empty:
                    required_cols = [
                        "Ticker",
                        "Raw_Score",
                        "Risk_Adj_Score",
                        "Volatility",
                    ]
                    if all(col in cluster_stocks.columns for col in required_cols):
                        top_raw = cluster_stocks.nlargest(8, "Raw_Score")[required_cols]

                        print(f"\nCluster {cluster_id} - Top 8 by Raw Score:")
                        print(top_raw.to_string(index=False))
                        cluster_avg_raw = cluster_performance.loc[
                            cluster_performance["Cluster_ID"] == cluster_id,
                            "Avg_Raw_Score",
                        ].values
                        cluster_avg_risk = cluster_performance.loc[
                            cluster_performance["Cluster_ID"] == cluster_id,
                            "Avg_Risk_Adj_Score",
                        ].values
                        if len(cluster_avg_raw) > 0:
                            print(f"Cluster Avg Raw Score: {cluster_avg_raw[0]:.2f}")
                        if len(cluster_avg_risk) > 0:
                            print(
                                f"Cluster Avg Risk Adj Score: {cluster_avg_risk[0]:.2f}"
                            )
                    else:
                        print(
                            f"\nCluster {cluster_id} - Missing required columns in detailed_clusters_df to show top stocks."
                        )
                else:
                    print(
                        f"\nCluster {cluster_id} - No stocks found in detailed_clusters_df for this cluster."
                    )
        else:
            print("\n=== TOP STOCKS BY RAW SCORE PER SELECTED CLUSTER ===")
            print(
                "Skipping - Detailed cluster information ('input_detailed_clusters_df') not found in the output dictionary."
            )

    else:
        print("\n=== SELECTED CLUSTERS ===")
        print("No clusters were selected based on the criteria.")

    print(f"\n=== FINAL SELECTED STOCKS (FILTERED & WEIGHTED) ===")
    if not selected_stocks.empty:
        print(
            "* Stocks actually selected based on Risk_Adj_Score (and optional thresholds) within each cluster."
        )
        print(
            "* Position weights assigned based on Risk_Adj_Score within the final selected portfolio."
        )

        desired_cols = [
            "Cluster_ID",
            "Ticker",
            "Raw_Score",
            "Risk_Adj_Score",
            "Volatility",
            "Weight",
            "Cluster_Avg_Raw_Score",
            "Cluster_Avg_Risk_Adj_Score",
        ]
        available_cols = [col for col in desired_cols if col in selected_stocks.columns]
        print(
            selected_stocks[available_cols]
            .sort_values(["Cluster_ID", "Risk_Adj_Score"], ascending=[True, False])
            .to_string(index=False)
        )

        print("\n=== PORTFOLIO SUMMARY ===")
        print(f"Total Stocks Selected: {len(selected_stocks)}")
        print(
            f"Average Raw Score: {selected_stocks.get('Raw_Score', pd.Series(dtype=float)).mean():.2f}"
        )
        print(
            f"Average Risk-Adjusted Score: {selected_stocks.get('Risk_Adj_Score', pd.Series(dtype=float)).mean():.2f}"
        )
        print(
            f"Average Volatility: {selected_stocks.get('Volatility', pd.Series(dtype=float)).mean():.2f}"
        )
        print(
            f"Total Weight (should be close to 1.0): {selected_stocks.get('Weight', pd.Series(dtype=float)).sum():.4f}"
        )
        print("\nCluster Distribution:")
        print(selected_stocks["Cluster_ID"].value_counts().to_string())
    else:
        print("No stocks were selected after applying all filters and criteria.")


#########################


def select_stocks_from_clusters(
    cluster_stats_df,
    detailed_clusters_df,
    select_top_n_clusters=3,
    max_selection_per_cluster=5,
    min_cluster_size=5,
    penalty_IntraCluster_Corr=0.3,
    date_str=None,
    min_raw_score=None,  # <-- Added argument
    min_risk_adj_score=None,
):  # <-- Added argument
    """
    Pipeline to select stocks from better performing clusters, with optional score thresholds.

    Parameters:
    - cluster_stats_df: DataFrame with cluster statistics.
    - detailed_clusters_df: DataFrame with detailed cluster information including
                            'Ticker', 'Cluster_ID', 'Raw_Score', 'Risk_Adj_Score', etc.
    - select_top_n_clusters: int, Number of top clusters to select (default=3).
    - max_selection_per_cluster: int, Max number of stocks to select from each cluster (default=5).
    - min_cluster_size: int, Minimum size for a cluster to be considered (default=5).
    - penalty_IntraCluster_Corr: float, Penalty weight for intra-cluster correlation in
        composite score (default=0.3).
    - date_str: str, Date string for tracking/parameter storage.
    - min_raw_score: float, optional (default=None)
        Minimum Raw_Score required for a stock to be considered for selection.
        If None, no threshold is applied based on Raw_Score.
    - min_risk_adj_score: float, optional (default=None)
        Minimum Risk_Adj_Score required for a stock to be considered for selection.
        If None, no threshold is applied based on Risk_Adj_Score.

    Returns:
    - dict: A dictionary containing:
        - 'selected_top_n_cluster_ids': List of top selected cluster IDs.
        - 'selected_stocks': DataFrame of selected stocks.
        - 'cluster_performance': DataFrame of selected cluster metrics.
        - 'parameters': Dictionary of the input parameters used.
    """

    # Store input parameters
    parameters = {
        "date_str": date_str,
        "select_top_n_clusters": select_top_n_clusters,
        "max_selection_per_cluster": max_selection_per_cluster,
        "min_cluster_size": min_cluster_size,
        "min_raw_score": min_raw_score,  # <-- Stored parameter
        "min_risk_adj_score": min_risk_adj_score,  # <-- Stored parameter
        "penalty_IntraCluster_Corr": penalty_IntraCluster_Corr,
    }

    # ===== 1. Filter and Rank Clusters =====
    qualified_clusters = cluster_stats_df[
        cluster_stats_df["Size"] >= min_cluster_size
    ].copy()
    if qualified_clusters.empty:
        print(
            f"Warning: No clusters met the minimum size criteria ({min_cluster_size})."
        )
        return {
            "selected_stocks": pd.DataFrame(),
            "cluster_performance": pd.DataFrame(),
            "parameters": parameters,
        }

    qualified_clusters["Composite_Cluster_Score"] = (
        1 - penalty_IntraCluster_Corr
    ) * qualified_clusters["Avg_Raw_Score"] + penalty_IntraCluster_Corr * (
        1 - qualified_clusters["Avg_IntraCluster_Corr"]
    )
    ranked_clusters = qualified_clusters.sort_values(
        "Composite_Cluster_Score", ascending=False
    )
    selected_clusters = ranked_clusters.head(select_top_n_clusters)
    cluster_ids = selected_clusters["Cluster_ID"].tolist()

    if not cluster_ids:
        print("Warning: No clusters were selected based on ranking.")
        return {
            "selected_stocks": pd.DataFrame(),
            "cluster_performance": selected_clusters,  # Return empty selected clusters df
            "parameters": parameters,
        }

    # ===== 2. Select Stocks from Each Cluster =====
    selected_stocks_list = []
    for cluster_id in cluster_ids:
        # Get all stocks for the current cluster
        cluster_stocks = detailed_clusters_df[
            detailed_clusters_df["Cluster_ID"] == cluster_id
        ].copy()

        # ===> Apply Threshold Filters <===
        if min_raw_score is not None:
            cluster_stocks = cluster_stocks[
                cluster_stocks["Raw_Score"] >= min_raw_score
            ]
        if min_risk_adj_score is not None:
            cluster_stocks = cluster_stocks[
                cluster_stocks["Risk_Adj_Score"] >= min_risk_adj_score
            ]
        # ===> End of Added Filters <===

        # Proceed only if stocks remain after filtering
        if len(cluster_stocks) > 0:
            # Sort remaining stocks by Risk_Adj_Score and select top N
            top_stocks = cluster_stocks.sort_values(
                "Risk_Adj_Score", ascending=False
            ).head(max_selection_per_cluster)

            # Add cluster-level metrics to the selected stock rows
            cluster_metrics = selected_clusters[
                selected_clusters["Cluster_ID"] == cluster_id
            ].iloc[0]
            for col in [
                "Composite_Cluster_Score",
                "Avg_IntraCluster_Corr",
                "Avg_Volatility",
                "Avg_Raw_Score",
                "Avg_Risk_Adj_Score",
                "Size",
            ]:  # Added Size for context
                # Use .get() for safety if a column might be missing
                top_stocks[f"Cluster_{col}"] = cluster_metrics.get(col, None)
            selected_stocks_list.append(top_stocks)

    # Consolidate selected stocks
    if selected_stocks_list:
        selected_stocks = pd.concat(selected_stocks_list)
        # Recalculate weights based on the final selection
        if selected_stocks["Risk_Adj_Score"].sum() != 0:
            selected_stocks["Weight"] = (
                selected_stocks["Risk_Adj_Score"]
                / selected_stocks["Risk_Adj_Score"].sum()
            )
        else:
            # Handle case where all selected scores are zero (unlikely but possible)
            selected_stocks["Weight"] = (
                1 / len(selected_stocks) if len(selected_stocks) > 0 else 0
            )

        selected_stocks = selected_stocks.sort_values(
            ["Cluster_ID", "Risk_Adj_Score"], ascending=[True, False]
        )
    else:
        selected_stocks = pd.DataFrame()
        print(
            "Warning: No stocks met selection criteria (including score thresholds if applied)."
        )

    # ===== 3. Prepare Enhanced Output Reports =====
    cluster_performance = selected_clusters.copy()
    # Calculate how many stocks were actually selected per cluster after filtering
    cluster_performance["Stocks_Selected"] = cluster_performance["Cluster_ID"].apply(
        lambda x: (
            len(selected_stocks[selected_stocks["Cluster_ID"] == x])
            if not selected_stocks.empty
            else 0
        )
    )

    if not selected_stocks.empty:
        # Ensure Avg_IntraCluster_Corr exists before calculating diversification
        if "Avg_IntraCluster_Corr" in cluster_performance.columns:
            cluster_performance["Intra_Cluster_Diversification"] = (
                1 - cluster_performance["Avg_IntraCluster_Corr"]
            )
        else:
            cluster_performance["Intra_Cluster_Diversification"] = pd.NA  # Or None
    else:
        # Handle case where selected_stocks is empty
        cluster_performance["Intra_Cluster_Diversification"] = pd.NA  # Or None

    # ===> Package results and parameters
    results_bundle = {
        "selected_top_n_cluster_ids": cluster_ids,
        "selected_stocks": selected_stocks,
        "cluster_performance": cluster_performance,
        "parameters": parameters,
    }

    return results_bundle


import pandas as pd
import numpy as np
import logging  # Assuming logging is set up elsewhere

# Define a small epsilon to prevent division by zero
EPSILON = 1e-9


def select_stocks_from_clusters_ai(
    cluster_stats_df,
    detailed_clusters_df,  # MUST contain 'Volatility' column if using 'InverseVolatility'
    # MUST contain 'Risk_Adj_Score' if using 'RiskAdjScore'
    select_top_n_clusters=3,
    max_selection_per_cluster=5,
    min_cluster_size=5,
    penalty_IntraCluster_Corr=0.3,
    weighting_scheme="RiskAdjScore",  # <-- Changed default back, added as option
    date_str=None,
    min_raw_score=None,
    min_risk_adj_score=None,
):
    """
    Pipeline to select stocks from better performing clusters, applying a specified
    weighting scheme.

    Parameters:
    - cluster_stats_df: DataFrame with cluster statistics.
    - detailed_clusters_df: DataFrame with detailed cluster information including
                            'Ticker', 'Cluster_ID', 'Raw_Score', 'Risk_Adj_Score',
                            and 'Volatility' (required for InverseVolatility).
    - select_top_n_clusters: int, Number of top clusters to select (default=3).
    - max_selection_per_cluster: int, Max number of stocks to select from each cluster (default=5).
    - min_cluster_size: int, Minimum size for a cluster to be considered (default=5).
    - penalty_IntraCluster_Corr: float, Penalty weight for intra-cluster correlation in
                                     composite score (default=0.3).
    - weighting_scheme: str, Method for assigning portfolio weights. Options:
                          'RiskAdjScore' (default, weights by Risk Adjusted Score),
                          'EqualWeight',
                          'InverseVolatility'.
    - date_str: str, Date string for tracking/parameter storage.
    - min_raw_score: float, optional (default=None)
        Minimum Raw_Score required for a stock to be considered for selection.
    - min_risk_adj_score: float, optional (default=None)
        Minimum Risk_Adj_Score required for a stock to be considered for selection.

    Returns:
    - dict: A dictionary containing:
        - 'selected_top_n_cluster_ids': List of top selected cluster IDs.
        - 'selected_stocks': DataFrame of selected stocks with weights based on scheme.
        - 'cluster_performance': DataFrame of selected cluster metrics.
        - 'parameters': Dictionary of the input parameters used.
    """
    # Validate weighting scheme input
    valid_schemes = [
        "RiskAdjScore",
        "EqualWeight",
        "InverseVolatility",
    ]  # Added RiskAdjScore
    if weighting_scheme not in valid_schemes:
        logging.warning(
            f"Invalid weighting_scheme '{weighting_scheme}'. "
            f"Defaulting to 'RiskAdjScore'. Valid options: {valid_schemes}"
        )
        weighting_scheme = "RiskAdjScore"  # Default if invalid input

    # Check if required columns exist for the chosen scheme
    required_col = None
    if weighting_scheme == "InverseVolatility":
        required_col = "Volatility"
    elif weighting_scheme == "RiskAdjScore":
        required_col = "Risk_Adj_Score"
        # Check if Risk_Adj_Score actually exists in the input, as it's crucial
        if required_col not in detailed_clusters_df.columns:
            logging.error(
                f"Weighting scheme '{weighting_scheme}' selected, but "
                f"required column '{required_col}' is missing in detailed_clusters_df input. "
                f"Cannot proceed with this scheme."
            )
            # Fallback or error - let's error more definitively here as it's the core input
            # Returning None results to signal failure
            parameters["error"] = (
                f"Missing required column '{required_col}' for scheme '{weighting_scheme}'"
            )
            return {
                "selected_top_n_cluster_ids": [],
                "selected_stocks": pd.DataFrame(),
                "cluster_performance": pd.DataFrame(),
                "parameters": parameters,
            }

    # Specific check for InverseVolatility if it's the chosen scheme
    if (
        weighting_scheme == "InverseVolatility"
        and required_col not in detailed_clusters_df.columns
    ):
        logging.error(
            f"Weighting scheme 'InverseVolatility' selected, but "
            f"'Volatility' column is missing in detailed_clusters_df input. "
            f"Cannot proceed with this scheme. Check the upstream analyze_clusters function."
        )
        logging.warning(
            "Falling back to 'EqualWeight' due to missing 'Volatility' column."
        )
        weighting_scheme = "EqualWeight"  # Fallback if Volatility is missing
        required_col = None  # Reset required_col as we've switched scheme

    # Store input parameters
    parameters = {
        "date_str": date_str,
        "select_top_n_clusters": select_top_n_clusters,
        "max_selection_per_cluster": max_selection_per_cluster,
        "min_cluster_size": min_cluster_size,
        "min_raw_score": min_raw_score,
        "min_risk_adj_score": min_risk_adj_score,
        "penalty_IntraCluster_Corr": penalty_IntraCluster_Corr,
        "weighting_scheme": weighting_scheme,  # Store the *actual* scheme used
    }

    # ===== 1. Filter and Rank Clusters (No Change) =====
    qualified_clusters = cluster_stats_df[
        cluster_stats_df["Size"] >= min_cluster_size
    ].copy()
    if qualified_clusters.empty:
        logging.warning(
            f"No clusters met the minimum size criteria ({min_cluster_size})."
        )
        return {
            "selected_top_n_cluster_ids": [],
            "selected_stocks": pd.DataFrame(),
            "cluster_performance": pd.DataFrame(),
            "parameters": parameters,
        }

    qualified_clusters["Composite_Cluster_Score"] = (
        1 - penalty_IntraCluster_Corr
    ) * qualified_clusters["Avg_Raw_Score"] + penalty_IntraCluster_Corr * (
        1 - qualified_clusters["Avg_IntraCluster_Corr"]
    )
    ranked_clusters = qualified_clusters.sort_values(
        "Composite_Cluster_Score", ascending=False
    )
    selected_clusters = ranked_clusters.head(select_top_n_clusters)
    cluster_ids = selected_clusters["Cluster_ID"].tolist()

    if not cluster_ids:
        logging.warning("No clusters were selected based on ranking.")
        return {
            "selected_top_n_cluster_ids": [],
            "selected_stocks": pd.DataFrame(),
            "cluster_performance": selected_clusters,
            "parameters": parameters,
        }

    # ===== 2. Select Stocks from Each Cluster (No Change in Selection Logic) =====
    selected_stocks_list = []
    for cluster_id in cluster_ids:
        cluster_stocks = detailed_clusters_df[
            detailed_clusters_df["Cluster_ID"] == cluster_id
        ].copy()

        # Apply Threshold Filters
        if min_raw_score is not None:
            cluster_stocks = cluster_stocks[
                cluster_stocks["Raw_Score"] >= min_raw_score
            ]
        if min_risk_adj_score is not None:
            cluster_stocks = cluster_stocks[
                cluster_stocks["Risk_Adj_Score"] >= min_risk_adj_score
            ]

        if len(cluster_stocks) > 0:
            # Selection still based on Risk_Adj_Score
            top_stocks = cluster_stocks.sort_values(
                "Risk_Adj_Score", ascending=False
            ).head(max_selection_per_cluster)

            # Add cluster-level metrics
            cluster_metrics = selected_clusters[
                selected_clusters["Cluster_ID"] == cluster_id
            ].iloc[0]
            required_metrics = [
                "Composite_Cluster_Score",
                "Avg_IntraCluster_Corr",
                "Avg_Volatility",
                "Avg_Raw_Score",
                "Avg_Risk_Adj_Score",
                "Size",
            ]
            for col in required_metrics:
                top_stocks[f"Cluster_{col}"] = cluster_metrics.get(col, None)
            selected_stocks_list.append(top_stocks)

    # Consolidate selected stocks
    if selected_stocks_list:
        selected_stocks = pd.concat(selected_stocks_list)

        # -------------------------------------------------------------
        # --- START MODIFICATION: Apply Selected Weighting Scheme ---
        # -------------------------------------------------------------
        num_selected = len(selected_stocks)
        logging.info(
            f"Applying '{weighting_scheme}' weighting to {num_selected} selected stocks."
        )

        if num_selected > 0:
            if weighting_scheme == "EqualWeight":
                equal_weight = 1.0 / num_selected
                selected_stocks["Weight"] = equal_weight

            elif weighting_scheme == "InverseVolatility":
                volatility = selected_stocks["Volatility"].copy()
                valid_vol_mask = volatility.notna() & (volatility > 0)
                num_invalid_vol = num_selected - valid_vol_mask.sum()
                if num_invalid_vol > 0:
                    logging.warning(
                        f"Found {num_invalid_vol} stocks with missing or non-positive "
                        f"volatility. They will receive zero weight in InverseVolatility scheme."
                    )
                    volatility.loc[~valid_vol_mask] = (
                        np.inf
                    )  # Set vol to inf -> inv_vol becomes 0

                inv_vol = 1.0 / volatility
                inv_vol = inv_vol.replace(
                    [np.inf, -np.inf], 0
                )  # Handle division by zero/inf

                total_inv_vol = inv_vol.sum()
                if total_inv_vol > EPSILON:
                    selected_stocks["Weight"] = inv_vol / total_inv_vol
                else:
                    logging.warning(
                        "Sum of inverse volatilities is near zero. Falling back to EqualWeight."
                    )
                    selected_stocks["Weight"] = 1.0 / num_selected

            # --- ADDED BACK: RiskAdjScore Weighting ---
            elif weighting_scheme == "RiskAdjScore":
                scores = selected_stocks["Risk_Adj_Score"].copy()
                # Handle potential negative scores if they should be excluded or floored at zero
                # For now, assume scores can be negative and sum might be zero or negative
                # If scores should only be positive, add: scores.loc[scores < 0] = 0
                total_score = scores.sum()

                if (
                    abs(total_score) > EPSILON
                ):  # Check if sum is significantly different from zero
                    # Normalize by the sum of scores
                    selected_stocks["Weight"] = scores / total_score
                    # Optional: Handle negative weights if needed (e.g., cap at 0, re-normalize positives)
                    # Example: if (selected_stocks['Weight'] < 0).any():
                    #     logging.warning("Negative weights generated by RiskAdjScore scheme. Capping at 0 and renormalizing.")
                    #     selected_stocks['Weight'] = np.maximum(selected_stocks['Weight'], 0)
                    #     selected_stocks['Weight'] /= selected_stocks['Weight'].sum()
                else:
                    logging.warning(
                        "Sum of Risk_Adj_Score is near zero. Cannot normalize weights. Falling back to EqualWeight."
                    )
                    selected_stocks["Weight"] = 1.0 / num_selected
            # --- END of RiskAdjScore Weighting ---

            # --- Placeholder for Future Schemes ---
            # elif weighting_scheme == 'MinimumVariance': ...
            # elif weighting_scheme == 'RiskParity': ...
            # -------------------------------------

        else:
            selected_stocks["Weight"] = 0
            logging.warning(
                "selected_stocks DataFrame became empty unexpectedly before weighting."
            )

        # --- Final check on weights ---
        if "Weight" in selected_stocks.columns:
            weight_sum = selected_stocks["Weight"].sum()
            if not np.isclose(weight_sum, 1.0):
                logging.warning(
                    f"Weights under scheme '{weighting_scheme}' do not sum close to 1.0 (Sum = {weight_sum:.6f}). This might indicate an issue (e.g., only negative scores)."
                )
                # Consider if re-normalization is needed depending on the scheme's intent

        # -----------------------------------------------------------
        # --- END MODIFICATION ---
        # -----------------------------------------------------------

        # Keep the sorting for consistent output
        selected_stocks = selected_stocks.sort_values(
            ["Cluster_ID", "Risk_Adj_Score"], ascending=[True, False]
        )
    else:
        selected_stocks = pd.DataFrame()
        logging.warning(
            "No stocks met selection criteria (including score thresholds if applied)."
        )

    # ===== 3. Prepare Enhanced Output Reports (No Change) =====
    cluster_performance = selected_clusters.copy()
    cluster_performance["Stocks_Selected"] = cluster_performance["Cluster_ID"].apply(
        lambda x: (
            len(selected_stocks[selected_stocks["Cluster_ID"] == x])
            if not selected_stocks.empty
            else 0
        )
    )

    if not selected_stocks.empty:
        if "Avg_IntraCluster_Corr" in cluster_performance.columns:
            cluster_performance["Intra_Cluster_Diversification"] = (
                1 - cluster_performance["Avg_IntraCluster_Corr"]
            )
        else:
            cluster_performance["Intra_Cluster_Diversification"] = pd.NA
    else:
        cluster_performance["Intra_Cluster_Diversification"] = pd.NA

    results_bundle = {
        "selected_top_n_cluster_ids": cluster_ids,
        "selected_stocks": selected_stocks,  # Contains weights from the chosen scheme
        "cluster_performance": cluster_performance,
        "parameters": parameters,
    }

    return results_bundle
