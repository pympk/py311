import sys
import os
import regutil
import time
import datetime
import numpy as np
import pandas as pd
import empyrical  
import warnings
import re
import json
import logging
import datetime
import pprint
import io
import traceback

from scipy.stats import zscore
from IPython.display import display, Markdown
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path




warnings.filterwarnings("ignore", message="Module \"zipline.assets\" not found.*")

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
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the Sharpe Ratio using empyrical (as pyfolio's is deprecated)
            sharpe_ratio = empyrical.sharpe_ratio(returns, risk_free=daily_risk_free_rate, annualization=days_per_year)

            # Calculate the Sortino Ratio using empyrical
            sortino_ratio = empyrical.sortino_ratio(returns, required_return=daily_risk_free_rate, annualization=days_per_year)

            # Calculate the Omega Ratio using empyrical
            omega_ratio = empyrical.omega_ratio(returns, risk_free=daily_risk_free_rate, annualization=days_per_year)

        n = len(returns) + 1  # Calculate n only once

        return {
            f"Sharpe {n}d": sharpe_ratio,
            f"Sortino {n}d": sortino_ratio,
            f"Omega {n}d": omega_ratio
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
        adj_close_prices = df.loc[ticker]['Adj Close'].sort_index()

        # Check if adj_close_prices is a Series
        if not isinstance(adj_close_prices, pd.Series):
             raise TypeError(f"Expected a Pandas Series for Adj Close prices of {ticker}. Check that {ticker} exists in the DataFrame, and that 'Adj Close' is a valid column")

        # Calculate returns
        returns_series = calculate_returns(adj_close_prices)

        if returns_series is not None:
            # Output debug data if requested
            if output_debug_data:
                print(f"--- Debug Data for {ticker} ---")
                print("\nAdj Close Prices (Dates and Values):")
                print(adj_close_prices)  #This is a Series, prints the index(dates) and values.
                print("\nReturns:")
                print(returns_series)

            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics(returns_series, risk_free_rate=risk_free_rate)

            if performance_metrics:
                # Create a DataFrame from the metrics
                metrics_df = pd.DataFrame(performance_metrics, index=[ticker])
                return metrics_df
            else:
                print(f"Could not calculate performance metrics for {ticker}.")
                return None  # Return None, not an empty DataFrame
        else:
            print(f"Failed to calculate returns for {ticker}.")
            return None # Return None, not an empty DataFrame

    except KeyError:
        print(f"Ticker '{ticker}' not found in DataFrame.")
        return None # Return None, not an empty DataFrame
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None # Return None, not an empty DataFrame


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


def filter_df_dates_to_reference_symbol_obsolete(df, reference_symbol="AAPL"):
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
        return pd.DataFrame()  # Return an empty DataFrame if reference_symbol is not found
    
    original_symbols = df.index.get_level_values('Symbol').unique().tolist()

    # Filter symbols based on date index matching with the reference symbol
    filtered_symbols = []
    for symbol in original_symbols:
        try: # Handle the case where a symbol might be missing from the df
            symbol_dates = df.loc[symbol].index
        except KeyError:
            continue # Skip to the next symbol if this one is missing

        if (len(symbol_dates) == len(reference_dates) and symbol_dates.equals(reference_dates)):
            filtered_symbols.append(symbol)

    # Create the filtered DataFrame
    df_filtered = df.loc[filtered_symbols]


    # Analyze the filtering results
    print(f"Original number of symbols: {len(original_symbols)}")
    print(f"Number of symbols after filtering: {len(filtered_symbols)}")
    print(f"Number of symbols filtered out: {len(original_symbols) - len(filtered_symbols)}")

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
            print(f"\nSymbol '{first_filtered_symbol}' not found in the original DataFrame.")


    print("\nFiltered DataFrame info:")
    print(df_filtered.info())

    return df_filtered


def filter_df_dates_to_reference_symbol(df, reference_symbol="AAPL"):
    """
    Filters symbols in a DataFrame based on date index matching a reference symbol
    and provides analysis of the filtering results. The function will adapt
    if the symbol index level is named 'Symbol' or 'Ticker'.

    Args:
        df (pd.DataFrame): DataFrame with a MultiIndex. The first level of the
                           MultiIndex should be the symbol identifier (expected to be
                           named 'Symbol' or 'Ticker'), and the second level should be 'Date'.
        reference_symbol (str): The symbol to use as the reference for date comparison.
                                Defaults to "AAPL".

    Returns:
        pd.DataFrame: The filtered DataFrame. Prints analysis to standard output.
                      Returns an empty DataFrame if errors occur (e.g., reference symbol
                      not found, or incompatible index name).
        filtered_out_symbols (list): List of symbols that were filtered out.              
    """

    if not isinstance(df.index, pd.MultiIndex) or len(df.index.levels) < 2:
        print("Error: DataFrame must have a MultiIndex with at least two levels.")
        return pd.DataFrame()

    # Determine the name of the symbol index level
    symbol_level_name = None
    if df.index.names[0] == 'Symbol':
        symbol_level_name = 'Symbol'
    elif df.index.names[0] == 'Ticker':
        symbol_level_name = 'Ticker'
    else:
        print(f"Error: The first level of the DataFrame's index must be named 'Symbol' or 'Ticker', but found '{df.index.names[0]}'.")
        return pd.DataFrame()

    # Get the date index for the reference symbol. Return empty DataFrame if symbol not found
    try:
        # Ensure reference_symbol is correctly accessed using the determined symbol_level_name
        # df.loc is smart enough to use the first level index if only one key is provided for the first level
        reference_dates = df.loc[reference_symbol].index
        if isinstance(reference_dates, pd.MultiIndex): # if reference_symbol still returns a multiindex (e.g. only one date)
            reference_dates = reference_dates.get_level_values('Date')

    except KeyError:
        print(f"Error: Reference symbol '{reference_symbol}' not found in DataFrame under index level '{symbol_level_name}'.")
        return pd.DataFrame()
    except AttributeError: # Handles cases where .index might not behave as expected if loc result is not a DataFrame
        print(f"Error: Could not retrieve date index for reference symbol '{reference_symbol}'.")
        return pd.DataFrame()


    original_symbols = df.index.get_level_values(symbol_level_name).unique().tolist()

    # Filter symbols based on date index matching with the reference symbol
    filtered_symbols = []
    for symbol_val in original_symbols:
        try: # Handle the case where a symbol might be missing or have issues
            # Access data for the current symbol using the determined symbol_level_name
            symbol_dates = df.loc[symbol_val].index
            if isinstance(symbol_dates, pd.MultiIndex): # if symbol_val still returns a multiindex
                 symbol_dates = symbol_dates.get_level_values('Date')

        except KeyError:
            print(f"Warning: Symbol '{symbol_val}' caused a KeyError during date extraction. Skipping.")
            continue # Skip to the next symbol if this one is missing or causes issues
        except AttributeError: # Handles cases where .index might not behave as expected
            print(f"Warning: Could not retrieve date index for symbol '{symbol_val}'. Skipping.")
            continue


        if len(symbol_dates) == len(reference_dates) and symbol_dates.equals(reference_dates):
            filtered_symbols.append(symbol_val)

    # Create the filtered DataFrame
    # df.loc can filter on the first level of a MultiIndex directly with a list of keys
    if filtered_symbols:
        df_filtered = df.loc[pd.IndexSlice[filtered_symbols, :]]
    else:
        df_filtered = pd.DataFrame() # Create an empty DataFrame with same columns if no symbols match
        if not df.empty:
            df_filtered = pd.DataFrame(columns=df.columns)
        if isinstance(df.index, pd.MultiIndex):
             df_filtered = df_filtered.set_index(df.index.names)
             # Ensure it's an empty df with the correct index structure if possible
             # This part might need adjustment based on how you want to handle an empty but structured df
             if not df_filtered.index.empty: # if set_index resulted in non-empty (e.g. from existing columns)
                 df_filtered = df_filtered.iloc[0:0]


    # Analyze the filtering results
    print(f"Using '{symbol_level_name}' as the symbol identifier.")
    print(f"Original number of {symbol_level_name}s: {len(original_symbols)}")
    print(f"Number of {symbol_level_name}s after filtering: {len(filtered_symbols)}")
    print(f"Number of {symbol_level_name}s filtered out: {len(original_symbols) - len(filtered_symbols)}")

    filtered_out_symbols = list(set(original_symbols) - set(filtered_symbols))

    print(f"\nFirst 10 {symbol_level_name}s that were filtered out:")
    print(filtered_out_symbols[:10])

    if filtered_out_symbols:
        print(f"\nExample of dates for first filtered out {symbol_level_name}:")
        first_filtered_symbol = filtered_out_symbols[0]
        try:
            symbol_data = df.loc[first_filtered_symbol]
            print(f"\nDates for {first_filtered_symbol}:")
            # Assuming the date level is consistently named 'Date' or is the second level
            if isinstance(symbol_data.index, pd.MultiIndex):
                print(symbol_data.index.get_level_values(df.index.names[1]))
            else:
                print(symbol_data.index)

        except KeyError:
            print(f"\n{symbol_level_name} '{first_filtered_symbol}' not found in the original DataFrame (this might happen if it was problematic).")


    print("\nFiltered DataFrame info:")
    if not df_filtered.empty:
        print(df_filtered.info())
    else:
        print("Filtered DataFrame is empty.")

    return df_filtered, filtered_out_symbols



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
        result = (
            df.groupby(level='Symbol', group_keys=False)
            .apply(lambda group: 
                # Sort group dates descending and take top N rows
                group.sort_index(level='Date', ascending=False).head(num)
            )
        )

        # Global sort for final output:
        # 1. Symbols in alphabetical order (ascending=True)
        # 2. Dates in chronological order (ascending=True) within each symbol
        result = result.sort_index(
            level=['Symbol', 'Date'], 
            ascending=[True, True]
        )

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
            filtered_data, 
            keys=kept_symbols, 
            names=['Symbol', 'Date']
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
        if suffix == 'M':
            # Already in millions, return directly
            return number
        elif suffix == 'K':
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


def extract_date_from_string(input_string, pattern=r'(\d{4}-\d{2}-\d{2})', group_index=1):
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
        raise ValueError(f"No match found for pattern '{pattern}' in input string: '{input_string}'")

    try:
        extracted_value = match.group(group_index)
        return extracted_value
    except IndexError:
        raise ValueError(f"Invalid group index: {group_index}. The pattern '{pattern}' does not have this many capturing groups.")


def get_matching_files(dir, create_dir=True, start_file_pattern='df_OHLCV_'):
    """Return list of files matching specified pattern in directory"""
    if create_dir:
        os.makedirs(dir, exist_ok=True)
    try:
        return [f for f in os.listdir(dir) if f.startswith(start_file_pattern)]
    except FileNotFoundError:
        return []


def process_downloads_dir(downloads_dir, limit=20, start_file_pattern='df_OHLCV_'):
    """Process Downloads directory with pattern-based filtering"""
    try:
        all_files = []
        for f in os.listdir(downloads_dir):
            file_path = os.path.join(downloads_dir, f)
            if os.path.isfile(file_path):
                all_files.append(file_path)
        
        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_files = all_files[:limit]
        matched_files = [f for f in latest_files 
                       if os.path.basename(f).startswith(start_file_pattern)]
        
        msg = (f"<span style='color:#00ffff;font-weight:500'>[Downloads] Scanned latest {len(latest_files)} files • "
               f"Found {len(matched_files)} '{start_file_pattern}' matches</span>")
        display(Markdown(msg))
        return matched_files
    
    except Exception as e:
        display(Markdown(f"<span style='color:red'>Error accessing Downloads: {str(e)}</span>"))
        return []


def display_file_selector(files_with_source, start_file_pattern):
    """Show interactive file selector with dynamic pattern"""
    display(Markdown(f"**Available '{start_file_pattern}' files:**"))
    
    for idx, (file_path, source) in enumerate(files_with_source, 1):
        name = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        timestamp = os.path.getmtime(file_path)
        
        size_mb = size / (1024 * 1024)
        formatted_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        
        file_info = (
            f"- ({idx}) `[{source.upper()}]` `{name}` "
            f"<span style='color:#00ffff'>"
            f"({size_mb:.2f} MB, {formatted_date})"
            f"</span>"
        )
        display(Markdown(file_info))
    
    print(f'\nInput a number to select file (1-{len(files_with_source)})')


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
                return files_with_source[choice-1][0]  # Return the path from tuple
            display(Markdown(f"<span style='color:red'>Enter 1-{len(files_with_source)}</span>"))
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
    cov_matrix = np.einsum('t,tij->ij', weights,
                          np.einsum('ti,tj->tij', demeaned.values, demeaned.values))

    if return_corr:
        # Handle zero variances to avoid division by zero
        variances = np.diag(cov_matrix).copy()
        variances[variances <= 0] = 1e-10  # Prevent NaN/Inf during normalization
        std_devs = np.sqrt(variances)

        # Calculate correlation matrix
        correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
        correlation_matrix = pd.DataFrame(correlation_matrix, index=df.columns, columns=df.columns)
    else:
        correlation_matrix = None

    cov_matrix = pd.DataFrame(cov_matrix, index=df.columns, columns=df.columns) if return_cov else None

    if not return_corr and not return_cov:
        return None

    return (cov_matrix, correlation_matrix) if return_cov and return_corr else (cov_matrix if return_cov else correlation_matrix)


def get_cov_corr_ewm_matrices_chunked(df, span=21, return_corr=True, return_cov=True, chunk_size=100):
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
            cov_chunk = np.dot(weighted_chunk_i.T, chunk_j)  # Shape (chunk_size, chunk_size)
            
            # Fill the covariance matrix
            cov_matrix[i:i_end, j:j_end] = cov_chunk
            
            # Fill symmetric part if not on diagonal
            if i != j:
                cov_matrix[j:j_end, i:i_end] = cov_chunk.T
    
    # Prepare results
    results = []
    
    if return_cov:
        cov_matrix_df = pd.DataFrame(cov_matrix, 
                                   index=clean_df.columns, 
                                   columns=clean_df.columns)
        results.append(cov_matrix_df)
    
    if return_corr:
        # Handle zero variances
        variances = np.diag(cov_matrix).copy()
        variances[variances <= 0] = 1e-10  # Small positive value
        std_devs = np.sqrt(variances)
        
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        corr_matrix_df = pd.DataFrame(corr_matrix,
                                    index=clean_df.columns,
                                    columns=clean_df.columns)
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


def main_processor(data_dir='../data', downloads_dir=None, downloads_limit=20,
                   clean_name_override=None, start_file_pattern='df_OHLCV_',
                   contains_pattern=None):
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
        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    elif downloads_dir == '': # Treat empty string as explicitly skipping downloads
        downloads_dir = None

    # Normalize data_dir path for consistent processing and get its base name
    data_dir_abs = os.path.abspath(data_dir)
    data_dir_label = os.path.basename(data_dir_abs) or data_dir_abs # Use basename, fallback to full path if basename is empty (e.g., root dir)

    # Get data directory files matching BOTH patterns
    data_files_raw = get_matching_files(data_dir_abs, create_dir=True, start_file_pattern=start_file_pattern)
    data_files = [(os.path.join(data_dir_abs, f), data_dir_label) # <-- Use base name as origin label
                  for f in data_files_raw
                  if not contains_pattern or contains_pattern in f] # Apply contains_pattern filter

    # Get downloads files matching BOTH patterns
    downloads_files = []
    # Only search downloads if the path exists and was provided (not None or '')
    if downloads_dir and os.path.exists(downloads_dir):
        raw_downloads = process_downloads_dir(downloads_dir, downloads_limit, start_file_pattern=start_file_pattern)
        # Apply contains_pattern filter to downloads files (checking basename)
        filtered_downloads = [f for f in raw_downloads
                              if not contains_pattern or contains_pattern in os.path.basename(f)]
        downloads_files = [(f, 'downloads') for f in filtered_downloads] # Keep 'downloads' label distinct
    elif downloads_dir and not os.path.exists(downloads_dir):
         display(Markdown(f"**Warning:** Downloads directory specified but not found: `{downloads_dir}`"))


    # Internal list still uses tuples for display logic
    ohlcv_files = data_files + downloads_files

    # Construct informative error/selection messages
    search_criteria = f"starting with '{start_file_pattern}'"
    if contains_pattern:
        search_criteria += f" and containing '{contains_pattern}'"

    # Create the list of basenames to be returned
    displayed_filenames = [os.path.basename(f_path) for f_path, _ in ohlcv_files]

    if not ohlcv_files: # Check the original tuple list
        display(Markdown(f"**Error:** No files found matching {search_criteria}! "
                         f"(Searched: '{data_dir_label}' dir and downloads (if applicable))"))
        # Return None, None, and an empty list for displayed_filenames
        return None, None, [] # Return empty list [] directly

    # Pass the tuple list to display (it needs the origin info)
    display_file_selector(ohlcv_files, search_criteria)
    selected_file = get_user_choice(ohlcv_files) # This returns the selected path or None

    if selected_file is None: # Handle case where user cancels
         display(Markdown(f"**Info:** No file selected."))
         # Return None, None, but DO return the list of filenames that *were* displayed
         return None, None, displayed_filenames

    # --- File was selected ---
    clean_name = generate_clean_filename(os.path.basename(selected_file))
    if clean_name_override is not None:
        clean_name = clean_name_override
    # Destination path still uses the full absolute path for reliability
    dest_path = os.path.join(data_dir_abs, clean_name)

    display(Markdown(f"""
    **Selected paths:**
    - Source: `{selected_file}`
    - Destination: `{dest_path}`
    """))

    # Return the selected file, destination path, AND the list of displayed *basenames*
    return selected_file, dest_path, displayed_filenames



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
    selected_stocks = output.get('selected_stocks', pd.DataFrame())
    cluster_performance = output.get('cluster_performance', pd.DataFrame())
    used_params = output.get('parameters', {})
    # Extract the input DataFrames needed for the report
    # cluster_stats_df = output.get('input_cluster_stats_df') # Might be None
    cluster_stats_df = output.get('cluster_stats_df') # Might be None
    # detailed_clusters_df = output.get('input_detailed_clusters_df') # Might be None
    detailed_clusters_df = output.get('detailed_clusters_df') # Might be None

    # --- Start of Original Code Block (adapted) ---

    print("\n=== CLUSTER SELECTION CRITERIA ===")
    print("* Using Composite_Cluster_Score (balancing Raw Score and diversification) for cluster ranking.")
    print("* Using Risk_Adj_Score for stock selection within clusters.")

    num_selected_clusters = len(cluster_performance) if not cluster_performance.empty else 0
    # Use the extracted cluster_stats_df
    total_clusters = len(cluster_stats_df) if cluster_stats_df is not None and not cluster_stats_df.empty else 'N/A'

    print(f"* Selected top {num_selected_clusters} clusters from {total_clusters} total initial clusters.") # Adjusted wording slightly
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
        display_cols_exist = [col for col in [
                                'Cluster_ID', 'Size', 'Avg_Raw_Score', 'Avg_Risk_Adj_Score',
                                'Avg_IntraCluster_Corr', 'Avg_Volatility', 'Composite_Cluster_Score',
                                'Stocks_Selected', 'Intra_Cluster_Diversification']
                                if col in cluster_performance.columns]
        print(cluster_performance[display_cols_exist].sort_values('Composite_Cluster_Score', ascending=False).to_string(index=False))

        # Print top 8 stocks by Raw_Score for each selected cluster
        # Check if detailed_clusters_df was successfully extracted
        if detailed_clusters_df is not None and not detailed_clusters_df.empty:
            print("\n=== TOP STOCKS BY RAW SCORE PER SELECTED CLUSTER ===")
            print("""* Volatility is the standard deviation of daily returns over the past 250 trading days (example context).
* Note: The stocks below are shown ranked by Raw_Score for analysis,
*       but actual selection within the cluster was based on Risk_Adj_Score.""")

            for cluster_id in cluster_performance['Cluster_ID']:
                cluster_stocks = detailed_clusters_df[detailed_clusters_df['Cluster_ID'] == cluster_id]
                if not cluster_stocks.empty:
                    required_cols = ['Ticker', 'Raw_Score', 'Risk_Adj_Score', 'Volatility']
                    if all(col in cluster_stocks.columns for col in required_cols):
                        top_raw = cluster_stocks.nlargest(8, 'Raw_Score')[required_cols]

                        print(f"\nCluster {cluster_id} - Top 8 by Raw Score:")
                        print(top_raw.to_string(index=False))
                        cluster_avg_raw = cluster_performance.loc[cluster_performance['Cluster_ID'] == cluster_id, 'Avg_Raw_Score'].values
                        cluster_avg_risk = cluster_performance.loc[cluster_performance['Cluster_ID'] == cluster_id, 'Avg_Risk_Adj_Score'].values
                        if len(cluster_avg_raw) > 0: print(f"Cluster Avg Raw Score: {cluster_avg_raw[0]:.2f}")
                        if len(cluster_avg_risk) > 0: print(f"Cluster Avg Risk Adj Score: {cluster_avg_risk[0]:.2f}")
                    else:
                        print(f"\nCluster {cluster_id} - Missing required columns in detailed_clusters_df to show top stocks.")
                else:
                    print(f"\nCluster {cluster_id} - No stocks found in detailed_clusters_df for this cluster.")
        else:
            print("\n=== TOP STOCKS BY RAW SCORE PER SELECTED CLUSTER ===")
            print("Skipping - Detailed cluster information ('input_detailed_clusters_df') not found in the output dictionary.")

    else:
        print("\n=== SELECTED CLUSTERS ===")
        print("No clusters were selected based on the criteria.")


    print(f"\n=== FINAL SELECTED STOCKS (FILTERED & WEIGHTED) ===")
    if not selected_stocks.empty:
        print("* Stocks actually selected based on Risk_Adj_Score (and optional thresholds) within each cluster.")
        print("* Position weights assigned based on Risk_Adj_Score within the final selected portfolio.")

        desired_cols = ['Cluster_ID', 'Ticker', 'Raw_Score', 'Risk_Adj_Score',
                        'Volatility', 'Weight',
                        'Cluster_Avg_Raw_Score', 'Cluster_Avg_Risk_Adj_Score']
        available_cols = [col for col in desired_cols if col in selected_stocks.columns]
        print(selected_stocks[available_cols].sort_values(['Cluster_ID', 'Risk_Adj_Score'],
                                                        ascending=[True, False]).to_string(index=False))

        print("\n=== PORTFOLIO SUMMARY ===")
        print(f"Total Stocks Selected: {len(selected_stocks)}")
        print(f"Average Raw Score: {selected_stocks.get('Raw_Score', pd.Series(dtype=float)).mean():.2f}")
        print(f"Average Risk-Adjusted Score: {selected_stocks.get('Risk_Adj_Score', pd.Series(dtype=float)).mean():.2f}")
        print(f"Average Volatility: {selected_stocks.get('Volatility', pd.Series(dtype=float)).mean():.2f}")
        print(f"Total Weight (should be close to 1.0): {selected_stocks.get('Weight', pd.Series(dtype=float)).sum():.4f}")
        print("\nCluster Distribution:")
        print(selected_stocks['Cluster_ID'].value_counts().to_string())
    else:
        print("No stocks were selected after applying all filters and criteria.")



def select_stocks_from_clusters(cluster_stats_df, detailed_clusters_df,
                                select_top_n_clusters=3, max_selection_per_cluster=5,
                                min_cluster_size=5, penalty_IntraCluster_Corr=0.3,
                                date_str=None,
                                min_raw_score=None, # <-- Added argument
                                min_risk_adj_score=None): # <-- Added argument
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
        'date_str': date_str,
        'select_top_n_clusters': select_top_n_clusters,
        'max_selection_per_cluster': max_selection_per_cluster,
        'min_cluster_size': min_cluster_size,
        'min_raw_score': min_raw_score,         # <-- Stored parameter
        'min_risk_adj_score': min_risk_adj_score, # <-- Stored parameter
        'penalty_IntraCluster_Corr': penalty_IntraCluster_Corr,
    }
    
    # ===== 1. Filter and Rank Clusters =====
    qualified_clusters = cluster_stats_df[cluster_stats_df['Size'] >= min_cluster_size].copy()
    if qualified_clusters.empty:
        print(f"Warning: No clusters met the minimum size criteria ({min_cluster_size}).")
        return {
            'selected_stocks': pd.DataFrame(),
            'cluster_performance': pd.DataFrame(),
            'parameters': parameters
        }

    qualified_clusters['Composite_Cluster_Score'] = (
        (1 - penalty_IntraCluster_Corr) * qualified_clusters['Avg_Raw_Score'] +
        penalty_IntraCluster_Corr * (1 - qualified_clusters['Avg_IntraCluster_Corr'])
    )
    ranked_clusters = qualified_clusters.sort_values('Composite_Cluster_Score', ascending=False)
    selected_clusters = ranked_clusters.head(select_top_n_clusters)
    cluster_ids = selected_clusters['Cluster_ID'].tolist()

    if not cluster_ids:
        print("Warning: No clusters were selected based on ranking.")
        return {
            'selected_stocks': pd.DataFrame(),
            'cluster_performance': selected_clusters, # Return empty selected clusters df
            'parameters': parameters
        }


    # ===== 2. Select Stocks from Each Cluster =====
    selected_stocks_list = []
    for cluster_id in cluster_ids:
        # Get all stocks for the current cluster
        cluster_stocks = detailed_clusters_df[detailed_clusters_df['Cluster_ID'] == cluster_id].copy()

        # ===> Apply Threshold Filters <===
        if min_raw_score is not None:
            cluster_stocks = cluster_stocks[cluster_stocks['Raw_Score'] >= min_raw_score]
        if min_risk_adj_score is not None:
            cluster_stocks = cluster_stocks[cluster_stocks['Risk_Adj_Score'] >= min_risk_adj_score]
        # ===> End of Added Filters <===

        # Proceed only if stocks remain after filtering
        if len(cluster_stocks) > 0:
            # Sort remaining stocks by Risk_Adj_Score and select top N
            top_stocks = cluster_stocks.sort_values('Risk_Adj_Score', ascending=False).head(max_selection_per_cluster)

            # Add cluster-level metrics to the selected stock rows
            cluster_metrics = selected_clusters[selected_clusters['Cluster_ID'] == cluster_id].iloc[0]
            for col in ['Composite_Cluster_Score', 'Avg_IntraCluster_Corr', 'Avg_Volatility',
                        'Avg_Raw_Score', 'Avg_Risk_Adj_Score', 'Size']: # Added Size for context
                # Use .get() for safety if a column might be missing
                top_stocks[f'Cluster_{col}'] = cluster_metrics.get(col, None)
            selected_stocks_list.append(top_stocks)

    # Consolidate selected stocks
    if selected_stocks_list:
        selected_stocks = pd.concat(selected_stocks_list)
        # Recalculate weights based on the final selection
        if selected_stocks['Risk_Adj_Score'].sum() != 0:
            selected_stocks['Weight'] = (selected_stocks['Risk_Adj_Score'] /
                                        selected_stocks['Risk_Adj_Score'].sum())
        else:
            # Handle case where all selected scores are zero (unlikely but possible)
            selected_stocks['Weight'] = 1 / len(selected_stocks) if len(selected_stocks) > 0 else 0

        selected_stocks = selected_stocks.sort_values(['Cluster_ID', 'Risk_Adj_Score'],
                                                    ascending=[True, False])
    else:
        selected_stocks = pd.DataFrame()
        print("Warning: No stocks met selection criteria (including score thresholds if applied).")


    # ===== 3. Prepare Enhanced Output Reports =====
    cluster_performance = selected_clusters.copy()
    # Calculate how many stocks were actually selected per cluster after filtering
    cluster_performance['Stocks_Selected'] = cluster_performance['Cluster_ID'].apply(
        lambda x: len(selected_stocks[selected_stocks['Cluster_ID'] == x]) if not selected_stocks.empty else 0)

    if not selected_stocks.empty:
        # Ensure Avg_IntraCluster_Corr exists before calculating diversification
        if 'Avg_IntraCluster_Corr' in cluster_performance.columns:
            cluster_performance['Intra_Cluster_Diversification'] = 1 - cluster_performance['Avg_IntraCluster_Corr']
        else:
            cluster_performance['Intra_Cluster_Diversification'] = pd.NA # Or None
    else:
        # Handle case where selected_stocks is empty
        cluster_performance['Intra_Cluster_Diversification'] = pd.NA # Or None

    # ===> Package results and parameters
    results_bundle = {
        'selected_top_n_cluster_ids': cluster_ids,
        'selected_stocks': selected_stocks,
        'cluster_performance': cluster_performance,
        'parameters': parameters
    }

    return results_bundle



import pandas as pd
import numpy as np
import logging      # Assuming logging is set up elsewhere

# Define a small epsilon to prevent division by zero
EPSILON = 1e-9

def select_stocks_from_clusters_ai(
    cluster_stats_df,
    detailed_clusters_df, # MUST contain 'Volatility' column if using 'InverseVolatility'
                          # MUST contain 'Risk_Adj_Score' if using 'RiskAdjScore'
    select_top_n_clusters=3,
    max_selection_per_cluster=5,
    min_cluster_size=5,
    penalty_IntraCluster_Corr=0.3,
    weighting_scheme='RiskAdjScore', # <-- Changed default back, added as option
    date_str=None,
    min_raw_score=None,
    min_risk_adj_score=None):
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
    valid_schemes = ['RiskAdjScore', 'EqualWeight', 'InverseVolatility'] # Added RiskAdjScore
    if weighting_scheme not in valid_schemes:
        logging.warning(f"Invalid weighting_scheme '{weighting_scheme}'. "
                        f"Defaulting to 'RiskAdjScore'. Valid options: {valid_schemes}")
        weighting_scheme = 'RiskAdjScore' # Default if invalid input

    # Check if required columns exist for the chosen scheme
    required_col = None
    if weighting_scheme == 'InverseVolatility':
        required_col = 'Volatility'
    elif weighting_scheme == 'RiskAdjScore':
        required_col = 'Risk_Adj_Score'
        # Check if Risk_Adj_Score actually exists in the input, as it's crucial
        if required_col not in detailed_clusters_df.columns:
            logging.error(f"Weighting scheme '{weighting_scheme}' selected, but "
                          f"required column '{required_col}' is missing in detailed_clusters_df input. "
                          f"Cannot proceed with this scheme.")
            # Fallback or error - let's error more definitively here as it's the core input
            # Returning None results to signal failure
            parameters['error'] = f"Missing required column '{required_col}' for scheme '{weighting_scheme}'"
            return {
                'selected_top_n_cluster_ids': [],
                'selected_stocks': pd.DataFrame(),
                'cluster_performance': pd.DataFrame(),
                'parameters': parameters
            }

    # Specific check for InverseVolatility if it's the chosen scheme
    if weighting_scheme == 'InverseVolatility' and required_col not in detailed_clusters_df.columns:
         logging.error(f"Weighting scheme 'InverseVolatility' selected, but "
                       f"'Volatility' column is missing in detailed_clusters_df input. "
                       f"Cannot proceed with this scheme. Check the upstream analyze_clusters function.")
         logging.warning("Falling back to 'EqualWeight' due to missing 'Volatility' column.")
         weighting_scheme = 'EqualWeight' # Fallback if Volatility is missing
         required_col = None # Reset required_col as we've switched scheme


    # Store input parameters
    parameters = {
        'date_str': date_str,
        'select_top_n_clusters': select_top_n_clusters,
        'max_selection_per_cluster': max_selection_per_cluster,
        'min_cluster_size': min_cluster_size,
        'min_raw_score': min_raw_score,
        'min_risk_adj_score': min_risk_adj_score,
        'penalty_IntraCluster_Corr': penalty_IntraCluster_Corr,
        'weighting_scheme': weighting_scheme # Store the *actual* scheme used
    }

    # ===== 1. Filter and Rank Clusters (No Change) =====
    qualified_clusters = cluster_stats_df[cluster_stats_df['Size'] >= min_cluster_size].copy()
    if qualified_clusters.empty:
        logging.warning(f"No clusters met the minimum size criteria ({min_cluster_size}).")
        return {
            'selected_top_n_cluster_ids': [],
            'selected_stocks': pd.DataFrame(),
            'cluster_performance': pd.DataFrame(),
            'parameters': parameters
        }

    qualified_clusters['Composite_Cluster_Score'] = (
        (1 - penalty_IntraCluster_Corr) * qualified_clusters['Avg_Raw_Score'] +
        penalty_IntraCluster_Corr * (1 - qualified_clusters['Avg_IntraCluster_Corr'])
    )
    ranked_clusters = qualified_clusters.sort_values('Composite_Cluster_Score', ascending=False)
    selected_clusters = ranked_clusters.head(select_top_n_clusters)
    cluster_ids = selected_clusters['Cluster_ID'].tolist()

    if not cluster_ids:
        logging.warning("No clusters were selected based on ranking.")
        return {
            'selected_top_n_cluster_ids': [],
            'selected_stocks': pd.DataFrame(),
            'cluster_performance': selected_clusters,
            'parameters': parameters
        }

    # ===== 2. Select Stocks from Each Cluster (No Change in Selection Logic) =====
    selected_stocks_list = []
    for cluster_id in cluster_ids:
        cluster_stocks = detailed_clusters_df[detailed_clusters_df['Cluster_ID'] == cluster_id].copy()

        # Apply Threshold Filters
        if min_raw_score is not None:
            cluster_stocks = cluster_stocks[cluster_stocks['Raw_Score'] >= min_raw_score]
        if min_risk_adj_score is not None:
            cluster_stocks = cluster_stocks[cluster_stocks['Risk_Adj_Score'] >= min_risk_adj_score]

        if len(cluster_stocks) > 0:
            # Selection still based on Risk_Adj_Score
            top_stocks = cluster_stocks.sort_values('Risk_Adj_Score', ascending=False).head(max_selection_per_cluster)

            # Add cluster-level metrics
            cluster_metrics = selected_clusters[selected_clusters['Cluster_ID'] == cluster_id].iloc[0]
            required_metrics = ['Composite_Cluster_Score', 'Avg_IntraCluster_Corr', 'Avg_Volatility',
                                'Avg_Raw_Score', 'Avg_Risk_Adj_Score', 'Size']
            for col in required_metrics:
                top_stocks[f'Cluster_{col}'] = cluster_metrics.get(col, None)
            selected_stocks_list.append(top_stocks)

    # Consolidate selected stocks
    if selected_stocks_list:
        selected_stocks = pd.concat(selected_stocks_list)

        # -------------------------------------------------------------
        # --- START MODIFICATION: Apply Selected Weighting Scheme ---
        # -------------------------------------------------------------
        num_selected = len(selected_stocks)
        logging.info(f"Applying '{weighting_scheme}' weighting to {num_selected} selected stocks.")

        if num_selected > 0:
            if weighting_scheme == 'EqualWeight':
                equal_weight = 1.0 / num_selected
                selected_stocks['Weight'] = equal_weight

            elif weighting_scheme == 'InverseVolatility':
                volatility = selected_stocks['Volatility'].copy()
                valid_vol_mask = volatility.notna() & (volatility > 0)
                num_invalid_vol = num_selected - valid_vol_mask.sum()
                if num_invalid_vol > 0:
                     logging.warning(f"Found {num_invalid_vol} stocks with missing or non-positive "
                                     f"volatility. They will receive zero weight in InverseVolatility scheme.")
                     volatility.loc[~valid_vol_mask] = np.inf # Set vol to inf -> inv_vol becomes 0

                inv_vol = 1.0 / volatility
                inv_vol = inv_vol.replace([np.inf, -np.inf], 0) # Handle division by zero/inf

                total_inv_vol = inv_vol.sum()
                if total_inv_vol > EPSILON:
                    selected_stocks['Weight'] = inv_vol / total_inv_vol
                else:
                    logging.warning("Sum of inverse volatilities is near zero. Falling back to EqualWeight.")
                    selected_stocks['Weight'] = 1.0 / num_selected

            # --- ADDED BACK: RiskAdjScore Weighting ---
            elif weighting_scheme == 'RiskAdjScore':
                scores = selected_stocks['Risk_Adj_Score'].copy()
                # Handle potential negative scores if they should be excluded or floored at zero
                # For now, assume scores can be negative and sum might be zero or negative
                # If scores should only be positive, add: scores.loc[scores < 0] = 0
                total_score = scores.sum()

                if abs(total_score) > EPSILON: # Check if sum is significantly different from zero
                     # Normalize by the sum of scores
                     selected_stocks['Weight'] = scores / total_score
                     # Optional: Handle negative weights if needed (e.g., cap at 0, re-normalize positives)
                     # Example: if (selected_stocks['Weight'] < 0).any():
                     #     logging.warning("Negative weights generated by RiskAdjScore scheme. Capping at 0 and renormalizing.")
                     #     selected_stocks['Weight'] = np.maximum(selected_stocks['Weight'], 0)
                     #     selected_stocks['Weight'] /= selected_stocks['Weight'].sum()
                else:
                     logging.warning("Sum of Risk_Adj_Score is near zero. Cannot normalize weights. Falling back to EqualWeight.")
                     selected_stocks['Weight'] = 1.0 / num_selected
            # --- END of RiskAdjScore Weighting ---


            # --- Placeholder for Future Schemes ---
            # elif weighting_scheme == 'MinimumVariance': ...
            # elif weighting_scheme == 'RiskParity': ...
            # -------------------------------------

        else:
            selected_stocks['Weight'] = 0
            logging.warning("selected_stocks DataFrame became empty unexpectedly before weighting.")

        # --- Final check on weights ---
        if 'Weight' in selected_stocks.columns:
             weight_sum = selected_stocks['Weight'].sum()
             if not np.isclose(weight_sum, 1.0):
                  logging.warning(f"Weights under scheme '{weighting_scheme}' do not sum close to 1.0 (Sum = {weight_sum:.6f}). This might indicate an issue (e.g., only negative scores).")
                  # Consider if re-normalization is needed depending on the scheme's intent

        # -----------------------------------------------------------
        # --- END MODIFICATION ---
        # -----------------------------------------------------------

        # Keep the sorting for consistent output
        selected_stocks = selected_stocks.sort_values(['Cluster_ID', 'Risk_Adj_Score'],
                                                    ascending=[True, False])
    else:
        selected_stocks = pd.DataFrame()
        logging.warning("No stocks met selection criteria (including score thresholds if applied).")


    # ===== 3. Prepare Enhanced Output Reports (No Change) =====
    cluster_performance = selected_clusters.copy()
    cluster_performance['Stocks_Selected'] = cluster_performance['Cluster_ID'].apply(
        lambda x: len(selected_stocks[selected_stocks['Cluster_ID'] == x]) if not selected_stocks.empty else 0)

    if not selected_stocks.empty:
        if 'Avg_IntraCluster_Corr' in cluster_performance.columns:
             cluster_performance['Intra_Cluster_Diversification'] = 1 - cluster_performance['Avg_IntraCluster_Corr']
        else:
             cluster_performance['Intra_Cluster_Diversification'] = pd.NA
    else:
        cluster_performance['Intra_Cluster_Diversification'] = pd.NA

    results_bundle = {
        'selected_top_n_cluster_ids': cluster_ids,
        'selected_stocks': selected_stocks, # Contains weights from the chosen scheme
        'cluster_performance': cluster_performance,
        'parameters': parameters
    }

    return results_bundle



def extract_date_from_string(text_to_search: str) -> Optional[str]:
    """
    Extracts the first valid YYYY-MM-DD date string found anywhere in the text.

    Uses regex to find the pattern and optionally validates the date's
    calendar correctness (e.g., rejects '2023-02-30').

    Args:
        text_to_search: The string to search within.

    Returns:
        The extracted and validated date string (e.g., '2025-04-24') if found,
        otherwise None. Returns None if the format matches but the date is
        not a valid calendar date.
    """
    # Regex pattern:
    # (\d{4}-\d{2}-\d{2}) - Captures a sequence of 4 digits, hyphen,
    #                       2 digits, hyphen, 2 digits.
    # No ^ anchor, so it can be anywhere in the string.
    pattern = r"(\d{4}-\d{2}-\d{2})"

    # Use re.search to find the first occurrence anywhere in the string
    match = re.search(pattern, text_to_search)

    if match:
        potential_date_str = match.group(1) # Extract the captured group

        # --- Optional but recommended: Validate if it's a valid date ---
        try:
            # Attempt to parse the extracted string as a date
            # datetime.strptime(potential_date_str, '%Y-%m-%d')
            datetime.datetime.strptime(potential_date_str, '%Y-%m-%d')
            # If parsing succeeds, it's a valid format AND a valid calendar date
            return potential_date_str
        except ValueError:
            # The pattern matched, but it's not a valid calendar date
            # (e.g., "2023-13-01" or "2023-02-30")
            # print(f"Warning: Found pattern '{potential_date_str}' but it's not a valid date.")
            return None # Treat invalid calendar dates as "not found"
    else:
        # Pattern was not found anywhere in the string
        return None



from pathlib import Path
import os # Used for os.path.expanduser to robustly find the home directory

def get_recent_files_in_directory_obsolete(
    prefix: str = '',
    extension: str = '',
    count: int = 10,
    directory_name: str = "Downloads"
) -> list[str]:
    """
    Reads the most recent files matching a specific prefix and extension
    from a specified subdirectory within the user's home directory.

    Args:
        prefix (str): The prefix the filenames must start with (e.g., 'report', 'ticker').
                      Defaults to an empty string, matching files without a specific prefix.
        extension (str): The file extension to filter by (e.g., 'csv', 'txt', 'log').
                         Include the extension only, no leading dot.
                         Defaults to an empty string, matching any extension.
        count (int): The maximum number of recent filenames to return.
        directory_name (str): The name of the subdirectory in the user's home
                              folder to search (e.g., "Downloads", "Documents").

    Returns:
        list[str]: A list of the most recent filenames matching the criteria,
                   sorted from most recent to oldest. Returns an empty list if no
                   matching files are found or the directory doesn't exist.
    """
    try:
        # 1. Get the user's home directory
        home_dir = Path.home() # Preferred modern way
        # Fallback for some environments if Path.home() is problematic:
        # home_dir = Path(os.path.expanduser('~'))

        # 2. Construct the path to the target directory
        target_dir = home_dir / directory_name
        print(f'target_dir: {target_dir}')  # Debugging line to show the target directory
        
        if not target_dir.is_dir():
            # print(f"Error: Directory '{target_dir}' not found.") # Optional: more verbose
            return []

        # 3. Find all files matching the pattern (prefix*.extension)
        #    We use glob for pattern matching.
        #    Construct the glob pattern based on prefix and extension.
        pattern = f"{prefix}*.{extension}" if extension else f"{prefix}*"

        candidate_files = [
            f for f in target_dir.glob(pattern)
            if f.is_file()
        ]

        if not candidate_files:
            # print(f"No files matching pattern '{pattern}' found in '{target_dir}'.") # Optional: more verbose
            return []

        # 4. Sort these files by modification time (most recent first)
        #    Path.stat().st_mtime gives the timestamp of the last modification.
        sorted_files = sorted(
            candidate_files,
            key=lambda f: f.stat().st_mtime,
            reverse=True  # True for most recent first
        )

        # 5. Get the top 'count' files and extract their names
        recent_filenames = [file.name for file in sorted_files[:count]]

        return recent_filenames

    except Exception as e:
        print(f"An error occurred: {e}")
        return []



#########################

DEFAULT_FILTERS_ST = {
    'min_price': 10.0,
    'min_avg_volume_m': 2.0,
    'min_roe_pct': 5.0,
    'max_debt_eq': 1.5
}

DEFAULT_SCORING_WEIGHTS_ST = {
    'rsi': 0.35,
    'change': 0.35,
    'rel_volume': 0.20,
    'volatility': 0.10
}

DEFAULT_INV_VOL_COL_ST = 'ATR/Price %'
def select_short_term_stocks_debug(
    df_finviz,
    # df_cov, # Kept for signature compatibility, but not used in provided logic
    n_select=20,
    filters=DEFAULT_FILTERS_ST, # Use recommended defaults
    scoring_weights=DEFAULT_SCORING_WEIGHTS_ST, # Use recommended defaults
    inv_vol_col_name=DEFAULT_INV_VOL_COL_ST) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]: # Updated return type hint
    """
    Selects stocks with potential for positive returns over the next 1-2 days,
    focusing on mean reversion and volume confirmation. Provides detailed output
    including scores and weights for Equal, Inverse Volatility, and Score-Weighted schemes.

    Uses recommended baseline parameters, but they should be validated via backtesting.

    Args:
        df_finviz (pd.DataFrame): DataFrame with stock metrics (must include columns
                                used in filters and scoring). Index should be Ticker.
        df_cov (pd.DataFrame): Covariance matrix (Index=Ticker, Columns=Ticker).
                               Currently unused in the provided logic but kept for signature.
        n_select (int): Number of top stocks to select *initially*. The actual number
                        selected might be lower if fewer stocks pass filters/scoring.
        filters (dict): Dictionary defining filter thresholds. Uses recommended defaults if not provided.
        scoring_weights (dict): Dictionary defining weights for scoring components. Uses recommended defaults if not provided.
        inv_vol_col_name (str): Column name in df_finviz to use for Inverse Volatility calculation. Uses recommended default if not provided.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - pd.DataFrame: DataFrame with selected tickers as index, including filter/score details
                            and 'Weight_EW', 'Weight_IV', 'Weight_SW' columns.
                            Returns empty DataFrame on failure or if no stocks pass filters/scoring.
            - pd.DataFrame: The DataFrame *after* filtering but *before* scoring/selection.
                            Returns empty DataFrame on failure.
            - Dict[str, Any]: A flat dictionary containing the parameters used for the selection
                              (e.g., {'n_select': 20, 'filter_min_price': 10.0,
                               'score_weight_rsi': 0.35, 'inv_vol_col_name': 'ATR/Price %'}).
    """
    # Store initial n_select value for parameters output
    initial_n_select = n_select

    # --- Log Parameters ---
    logging.info("--- Starting Short-Term Stock Selection (Debug Mode) ---")
    logging.info(f"Initial Parameters: n_select={initial_n_select}")
    logging.info(f"Filters Used: {filters}")
    logging.info(f"Scoring Weights Used: {scoring_weights}")
    logging.info(f"Inverse Volatility Column Used: '{inv_vol_col_name}'")

    # --- Basic Input Validation ---
    if not isinstance(df_finviz, pd.DataFrame) or df_finviz.empty:
        logging.error("Input df_finviz is not a valid DataFrame or is empty.")
        # Create the parameters dict even on failure for consistent return signature
        parameters_used = {
            'n_select_requested': initial_n_select,
            'inv_vol_col_name': inv_vol_col_name
        }
        if isinstance(filters, dict):
            for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
        if isinstance(scoring_weights, dict):
            for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        return pd.DataFrame(), pd.DataFrame(), parameters_used

    if not isinstance(filters, dict) or not isinstance(scoring_weights, dict):
        logging.error("Filters and scoring_weights must be dictionaries.")
        parameters_used = {
            'n_select_requested': initial_n_select,
            'inv_vol_col_name': inv_vol_col_name
        }
        # Attempt to add parameters even if types are wrong, might fail if not iterable
        try:
            if isinstance(filters, dict):
                 for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
            if isinstance(scoring_weights, dict):
                 for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        except: pass # Ignore errors during parameter gathering on failure
        return pd.DataFrame(), pd.DataFrame(), parameters_used


    if abs(sum(scoring_weights.values()) - 1.0) > EPSILON:
         logging.warning(f"Scoring weights provided do not sum to 1.0 (Sum={sum(scoring_weights.values())}). Proceeding, but normalization might be affected.")

    df = df_finviz.copy()
    df_after_filter = pd.DataFrame() # Initialize for return

    # --- [Existing code sections 1 through 6 remain unchanged] ---
    # 1. Define Required Columns based on Inputs
    filter_cols = ['Price', 'Avg Volume, M', 'ROE %', 'Debt/Eq']
    score_input_cols = ['RSI', 'Change %', 'Rel Volume', 'ATR/Price %']
    inv_vol_req_col = [inv_vol_col_name] if inv_vol_col_name else []
    required_cols = list(set(filter_cols + score_input_cols + inv_vol_req_col))
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns in df_finviz: {missing_cols}. Required based on filters/weights/inv_vol_col: {required_cols}")
        parameters_used = {
            'n_select_requested': initial_n_select, 'inv_vol_col_name': inv_vol_col_name
        }
        if isinstance(filters, dict):
            for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
        if isinstance(scoring_weights, dict):
            for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        return pd.DataFrame(), pd.DataFrame(), parameters_used

    # 2. Data Preparation and Cleaning
    logging.debug("Converting required columns to numeric...")
    for col in required_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    initial_count = len(df)
    df.dropna(subset=required_cols, inplace=True)
    cleaned_count = len(df)
    logging.info(f"Cleaned data: Removed {initial_count - cleaned_count} rows with NaNs in essential columns ({required_cols}). {cleaned_count} remaining.")
    if cleaned_count == 0:
        logging.warning("No stocks remaining after NaN cleaning.")
        parameters_used = {
            'n_select_requested': initial_n_select, 'inv_vol_col_name': inv_vol_col_name
        }
        if isinstance(filters, dict):
            for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
        if isinstance(scoring_weights, dict):
            for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        return pd.DataFrame(), pd.DataFrame(), parameters_used

    # 3. Filtering
    logging.info("Applying filters...")
    try:
        filter_mask = pd.Series(True, index=df.index)
        if 'min_price' in filters: filter_mask &= (df['Price'] >= filters['min_price'])
        if 'min_avg_volume_m' in filters: filter_mask &= (df['Avg Volume, M'] >= filters['min_avg_volume_m'])
        if 'min_roe_pct' in filters: filter_mask &= (df['ROE %'] >= filters['min_roe_pct'])
        if 'max_debt_eq' in filters: filter_mask &= (df['Debt/Eq'] <= filters['max_debt_eq'])
        df_filtered = df[filter_mask].copy()
        df_after_filter = df_filtered.copy()
        filtered_count = len(df_filtered)
        logging.info(f"Filtering complete: {filtered_count} stocks passed filters.")
    except KeyError as e:
        logging.error(f"Filtering failed. Check keys. Missing column/key: {e}. Filters: {filters}")
        parameters_used = {
            'n_select_requested': initial_n_select, 'inv_vol_col_name': inv_vol_col_name
        }
        if isinstance(filters, dict):
            for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
        if isinstance(scoring_weights, dict):
            for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        return pd.DataFrame(), pd.DataFrame(), parameters_used # Return empty dfs, params
    except Exception as e:
         logging.error(f"An unexpected error occurred during filtering: {e}")
         parameters_used = {
             'n_select_requested': initial_n_select, 'inv_vol_col_name': inv_vol_col_name
         }
         if isinstance(filters, dict):
             for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
         if isinstance(scoring_weights, dict):
             for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
         return pd.DataFrame(), pd.DataFrame(), parameters_used

    if filtered_count == 0:
        logging.warning("No stocks passed the filtering criteria.")
        parameters_used = {
            'n_select_requested': initial_n_select, 'inv_vol_col_name': inv_vol_col_name
        }
        if isinstance(filters, dict):
            for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
        if isinstance(scoring_weights, dict):
            for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        return pd.DataFrame(), df_after_filter, parameters_used # Return empty selection, filtered df, params
    elif filtered_count < n_select:
        logging.warning(f"Only {filtered_count} stocks passed filters, less than n_select ({n_select}). Selecting all {filtered_count}.")
        n_select = filtered_count # Adjust n_select

    # 4. Scoring
    logging.info("Calculating component scores (Z-scores)...")
    try:
        z_rsi = z_score_series(df_filtered['RSI']).rename('z_RSI')
        z_change = z_score_series(df_filtered['Change %']).rename('z_Change%')
        z_rel_volume = z_score_series(df_filtered['Rel Volume']).rename('z_RelVolume')
        z_volatility = z_score_series(df_filtered['ATR/Price %']).rename('z_ATR/Price%')
        final_score = (
            z_rsi * scoring_weights.get('rsi', 0) * (-1) +
            z_change * scoring_weights.get('change', 0) * (-1) +
            z_rel_volume * scoring_weights.get('rel_volume', 0) * (1) +
            z_volatility * scoring_weights.get('volatility', 0) * (-1)
        ).rename('final_score').fillna(0)
    except KeyError as e:
        logging.error(f"Scoring failed. Missing column/key: {e}. Weights: {scoring_weights}")
        parameters_used = {
            'n_select_requested': initial_n_select, 'inv_vol_col_name': inv_vol_col_name
        }
        if isinstance(filters, dict):
            for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
        if isinstance(scoring_weights, dict):
            for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        return pd.DataFrame(), df_after_filter, parameters_used
    except Exception as e:
         logging.error(f"An unexpected error occurred during scoring: {e}")
         parameters_used = {
             'n_select_requested': initial_n_select, 'inv_vol_col_name': inv_vol_col_name
         }
         if isinstance(filters, dict):
             for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
         if isinstance(scoring_weights, dict):
             for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
         return pd.DataFrame(), df_after_filter, parameters_used

    # 5. Combine Data & Rank
    df_debug = pd.concat([
        df_filtered[required_cols], z_rsi, z_change, z_rel_volume, z_volatility, final_score
    ], axis=1)
    logging.info(f"Top 5 Stocks based on Intermediate Scores:\n{df_debug.sort_values('final_score', ascending=False).head(5)}")

    # 6. Selection
    logging.info(f"Ranking stocks by final_score and selecting top {n_select}...")
    n_select = min(n_select, len(df_debug)) # Ensure n_select is not > available
    if n_select == 0:
        logging.warning("No stocks available for selection after scoring/ranking.")
        parameters_used = {
            'n_select_requested': initial_n_select, 'n_select_actual': 0, 'inv_vol_col_name': inv_vol_col_name
        }
        if isinstance(filters, dict):
            for k, v in filters.items(): parameters_used[f'filter_{k}'] = v
        if isinstance(scoring_weights, dict):
            for k, v in scoring_weights.items(): parameters_used[f'score_weight_{k}'] = v
        return pd.DataFrame(), df_after_filter, parameters_used

    df_ranked = df_debug.sort_values('final_score', ascending=False)
    df_selected = df_ranked.head(n_select).copy()
    actual_n_select = len(df_selected) # Store the actual number selected

    # --- 7. Weighting (Unchanged Logic) ---
    logging.info(f"Applying 'EqualWeight', 'InverseVolatility', and 'ScoreWeighted' schemes...")
    # -- Equal Weight (EW) --
    df_selected['Weight_EW'] = 1.0 / actual_n_select if actual_n_select > 0 else 0
    # -- Inverse Volatility Weight (IV) --
    if not inv_vol_col_name or inv_vol_col_name not in df_selected.columns:
        df_selected['Weight_IV'] = np.nan
    else:
        volatility = pd.to_numeric(df_selected[inv_vol_col_name], errors='coerce')
        valid_vol_mask = volatility.notna() & (volatility > EPSILON)
        volatility.loc[~valid_vol_mask] = np.inf
        inv_vol = (1.0 / volatility).replace([np.inf, -np.inf], 0).fillna(0)
        total_inv_vol = inv_vol.sum()
        if total_inv_vol > EPSILON: df_selected['Weight_IV'] = inv_vol / total_inv_vol
        else: df_selected['Weight_IV'] = df_selected['Weight_EW'] # Fallback
    # -- Score Weighted (SW) --
    scores = pd.to_numeric(df_selected['final_score'], errors='coerce').fillna(0)
    min_score = scores.min()
    if min_score < 0: scores = scores - min_score # Shift to non-negative
    total_score = scores.sum()
    if abs(total_score) > EPSILON: df_selected['Weight_SW'] = scores / total_score
    else: df_selected['Weight_SW'] = df_selected['Weight_EW'] # Fallback

    # --- 8. Final Checks (Unchanged Logic) ---
    weight_cols = ['Weight_EW', 'Weight_IV', 'Weight_SW']
    for w_col in weight_cols:
        # ... (sum checks remain the same) ...
        if w_col in df_selected.columns and pd.api.types.is_numeric_dtype(df_selected[w_col]):
             weight_sum = df_selected[w_col].sum()
             if pd.isna(weight_sum): logging.warning(f"Weight sum for '{w_col}' resulted in NaN.")
             elif not np.isclose(weight_sum, 1.0, atol=EPSILON): logging.warning(f"Final weights for '{w_col}' do not sum close to 1.0 (Sum = {weight_sum:.6f}). Check calculations.")
        elif w_col in df_selected.columns: logging.warning(f"Weight column '{w_col}' exists but is not numeric.")


    logging.info(f"Selected Stocks ({actual_n_select} stocks) with Scores and Weights (Top {min(5, actual_n_select)} shown):\n{df_selected.head(min(5, actual_n_select))}")


    # --- 9. Prepare Parameters Dictionary for Output ---
    parameters_used = {
        'n_select_requested': initial_n_select, # The n_select passed into the function
        'n_select_actual': actual_n_select,     # The number actually selected (might be lower)
        'inv_vol_col_name': inv_vol_col_name
    }
    # Flatten filters into the dictionary
    if isinstance(filters, dict):
        for k, v in filters.items():
            parameters_used[f'filter_{k}'] = v
    # Flatten scoring_weights into the dictionary
    if isinstance(scoring_weights, dict):
        for k, v in scoring_weights.items():
            parameters_used[f'score_weight_{k}'] = v

    logging.info("--- Short-Term Stock Selection Finished ---")


    # --- 10. Return Results ---
    return df_selected, df_after_filter, parameters_used


def save_selection_results(
    df_selected: pd.DataFrame,
    parameters_used: Dict[str, Any],
    base_filepath: str,
    save_csv: bool = False # Option to also save as CSV for inspection
    ) -> bool:
    """
    Saves the selected stocks DataFrame and parameters dictionary to files.

    Saves DataFrame primarily as Parquet and parameters as JSON.
    Optionally saves DataFrame as CSV as well.

    Args:
        df_selected (pd.DataFrame): The DataFrame containing selected stocks and details.
        parameters_used (Dict[str, Any]): The flat dictionary of parameters used.
        base_filepath (str): The base path and filename *without* extension.
                            Example: 'results/selection_run_20231027'
        save_csv (bool, optional): If True, also saves the DataFrame as a CSV file.
                                  Defaults to False.

    Returns:
        bool: True if all intended save operations were successful, False otherwise.
    """
    success = True
    parquet_file = f"{base_filepath}.parquet"
    json_file = f"{base_filepath}_params.json"
    csv_file = f"{base_filepath}.csv"

    # Ensure the directory exists
    try:
        output_dir = os.path.dirname(base_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory for {base_filepath}: {e}")
        return False # Cannot proceed if directory creation fails

    # --- Save DataFrame as Parquet ---
    try:
        if not df_selected.empty:
            df_selected.to_parquet(parquet_file, engine='pyarrow', compression='zstd', index=True)
            logging.info(f"Successfully saved selected stocks DataFrame to Parquet: {parquet_file}")
        else:
            logging.warning(f"Selected stocks DataFrame is empty. Parquet file not saved: {parquet_file}")
            # We might consider this a partial success depending on requirements
            # Set success = False if saving an empty df isn't acceptable
    except Exception as e:
        logging.error(f"Failed to save DataFrame to Parquet ({parquet_file}): {e}")
        success = False

    # --- Optionally save DataFrame as CSV ---
    if save_csv:
        try:
            if not df_selected.empty:
                df_selected.to_csv(csv_file, index=True) # IMPORTANT: Save the index (Tickers)
                logging.info(f"Successfully saved selected stocks DataFrame to CSV: {csv_file}")
            else:
                logging.warning(f"Selected stocks DataFrame is empty. CSV file not saved: {csv_file}")
        except Exception as e:
            logging.error(f"Failed to save DataFrame to CSV ({csv_file}): {e}")
            success = False # Treat failure to save optional CSV as overall failure if desired

    # --- Save Parameters as JSON ---
    try:
        with open(json_file, 'w') as f:
            json.dump(parameters_used, f, indent=4) # Use indent for readability
        logging.info(f"Successfully saved parameters to JSON: {json_file}")
    except TypeError as e:
         logging.error(f"Failed to serialize parameters to JSON ({json_file}). Check for non-serializable types: {e}")
         success = False
    except Exception as e:
        logging.error(f"Failed to save parameters to JSON ({json_file}): {e}")
        success = False

    return success


def load_selection_results(
    base_filepath: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Loads the selected stocks DataFrame and parameters dictionary from files.

    Attempts to load DataFrame from Parquet first, then falls back to CSV if specified.
    Loads parameters from JSON.

    Args:
        base_filepath (str): The base path and filename *without* extension,
                            matching the one used in save_selection_results.
                            Example: 'results/selection_run_20231027'

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
            A tuple containing:
            - The loaded DataFrame (or None if loading failed or file not found).
            - The loaded parameters dictionary (or None if loading failed or file not found).
    """
    df_selected = None
    parameters_used = None

    parquet_file = f"{base_filepath}.parquet"
    json_file = f"{base_filepath}_params.json"
    csv_file = f"{base_filepath}.csv" # For fallback

    # --- Load DataFrame ---
    # Prioritize Parquet
    if os.path.exists(parquet_file):
        try:
            df_selected = pd.read_parquet(parquet_file, engine='pyarrow')
            logging.info(f"Successfully loaded DataFrame from Parquet: {parquet_file}")
        except Exception as e:
            logging.error(f"Failed to load DataFrame from Parquet ({parquet_file}): {e}")
            # Optionally, try CSV here if Parquet fails to load but exists
            # For now, we just log error and proceed to check CSV explicitly below

    # Fallback to CSV if Parquet doesn't exist (or failed loading - though we don't explicitly retry here)
    if df_selected is None and os.path.exists(csv_file):
        logging.info(f"Parquet file ({parquet_file}) not found or failed to load. Attempting to load from CSV: {csv_file}")
        try:
            # IMPORTANT: Use index_col=0 to read the first column as the index
            df_selected = pd.read_csv(csv_file, index_col=0)
            logging.info(f"Successfully loaded DataFrame from CSV: {csv_file}")
        except Exception as e:
            logging.error(f"Failed to load DataFrame from CSV ({csv_file}): {e}")
    elif df_selected is None:
        logging.warning(f"Could not find Parquet ({parquet_file}) or CSV ({csv_file}) for DataFrame.")


    # --- Load Parameters from JSON ---
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                parameters_used = json.load(f)
            logging.info(f"Successfully loaded parameters from JSON: {json_file}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from file ({json_file}): {e}")
        except Exception as e:
            logging.error(f"Failed to load parameters from JSON ({json_file}): {e}")
    else:
        logging.warning(f"Parameters JSON file not found: {json_file}")


    # --- Final Check ---
    if df_selected is None and parameters_used is None:
        logging.error(f"Failed to load both DataFrame and Parameters for base path: {base_filepath}")

    return df_selected, parameters_used


import pandas as pd
from typing import List, Optional

def add_columns_from_source(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    cols_to_add: List[str],
    match_col_base: Optional[str] = None, # Default to None now
    match_on_base_index: bool = False   # New flag to match on base index
) -> pd.DataFrame:
    """
    Adds specified columns from a source DataFrame to a base DataFrame,
    placing the added columns to the left of the original base columns.

    Rows are matched based on either a specified column in the base DataFrame
    OR the index of the base DataFrame, against the *index* of the source
    DataFrame.

    Args:
        base_df (pd.DataFrame):
            The DataFrame to which the columns will be added.
        source_df (pd.DataFrame):
            The DataFrame containing the column data to add. Its index MUST
            contain the keys for matching (e.g., Tickers).
        cols_to_add (List[str]):
            A list of string names of the columns in `source_df` to retrieve data from.
        match_col_base (Optional[str], optional):
            The string name of the column in `base_df` containing the keys
            (e.g., Tickers) used for matching against the `source_df` index.
            Required if `match_on_base_index` is False. Defaults to None.
        match_on_base_index (bool, optional):
            If True, use the index of `base_df` for matching against the
            `source_df` index. If False (default), use the column specified
            by `match_col_base`. Defaults to False.

    Returns:
        pd.DataFrame: A new DataFrame, with the specified `cols_to_add` from
                      `source_df` appearing first (on the left), followed by
                      all columns from the original `base_df`.
                      Values are mapped based on the specified matching criteria.
                      Returns NaN for the added columns where no match is found.

    Raises:
        KeyError: If `match_on_base_index` is False and `match_col_base` is None
                  or not in `base_df.columns`.
                  If any column in `cols_to_add` is not in `source_df.columns`.
        ValueError: If `cols_to_add` is empty, or if both `match_col_base`
                    is provided and `match_on_base_index` is True (ambiguous).
    """
    # --- Input validation ---
    if not cols_to_add:
        raise ValueError("The 'cols_to_add' list cannot be empty.")

    if match_on_base_index and match_col_base is not None:
        raise ValueError("Cannot specify both 'match_col_base' and 'match_on_base_index=True'. Choose one matching method.")
    if not match_on_base_index and match_col_base is None:
         raise ValueError("Must specify 'match_col_base' if 'match_on_base_index' is False.")

    if not match_on_base_index:
        if match_col_base not in base_df.columns:
            raise KeyError(f"Matching column '{match_col_base}' not found in base_df columns: {base_df.columns.tolist()}")
        match_key_description = f"column '{match_col_base}'"
    else:
        match_key_description = "index" # For warning messages


    missing_source_cols = [col for col in cols_to_add if col not in source_df.columns]
    if missing_source_cols:
        raise KeyError(f"Columns to add {missing_source_cols} not found in source_df columns: {source_df.columns.tolist()}")

    # --- Perform the merge ---
    # Select only the columns needed from the source to avoid unnecessary merging
    source_subset = source_df[cols_to_add]

    # Store original base columns for later reordering
    original_base_columns = base_df.columns.tolist()

    # Build arguments for pd.merge dynamically
    merge_kwargs = {
        'right': source_subset,
        'right_index': True,      # Always merge on the source's index
        'how': 'left',            # Keep all rows from base_df
        'suffixes': ('', '_source') # Add suffix if col name conflict occurs
    }

    if match_on_base_index:
        merge_kwargs['left_index'] = True # Use base_df's index
    else:
        merge_kwargs['left_on'] = match_col_base # Use specified base_df column

    # Execute the merge
    merged_df = pd.merge(
        base_df,
        **merge_kwargs
    )

    # --- Reorder columns ---
    # Identify the actual names of the columns added (handling potential suffixes)
    all_merged_cols = merged_df.columns.tolist()
    # Columns are considered "added" if they were NOT in the original base_df
    added_cols_actual_names = [col for col in all_merged_cols if col not in original_base_columns]

    # Create the desired final column order
    new_column_order = added_cols_actual_names + original_base_columns

    # Apply the new order
    # Use reindex to ensure all expected columns are present, handling potential edge cases
    final_df = merged_df.reindex(columns=new_column_order)


    # --- Optional: Add warnings for NaNs in added columns ---
    # Use the actual names of the added columns after merge for checking
    total_rows = len(final_df)
    for col_name in added_cols_actual_names:
        # Check if column exists (it should, due to reindex)
        if col_name in final_df.columns:
            num_nas = final_df[col_name].isna().sum()
            if num_nas > 0:
                print(f"Warning: {num_nas}/{total_rows} entries for added column '{col_name}' are NaN.")
                if num_nas == total_rows:
                     print(f"   -> Check if values in the base_df {match_key_description} exist in the index of the source DataFrame.")
        else:
             # This case should ideally not happen with the reindex approach
             print(f"Warning: Expected added column '{col_name}' was not found in the final reordered DataFrame.")


    return final_df


def z_score_series(series: pd.Series) -> pd.Series:
    """Calculates Z-score for a pandas Series, handling NaNs."""
    # Ensure input is numeric before zscoring
    numeric_series = pd.to_numeric(series, errors='coerce')
    if numeric_series.isnull().all(): # Handle case where all values are NaN after coercion
        return pd.Series(np.nan, index=series.index).rename(f"z_{series.name}")
    return pd.Series(zscore(numeric_series, nan_policy='omit'), index=series.index).rename(f"z_{series.name}")



# =================================================================================
# NEW ADDITIONS FOR BACKTESTING ENGINE - APPEND TO src/utils.py
# =================================================================================

import logging
import datetime
import json
import pprint
import io
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# --- Backtesting Setup and File Handling ---

def setup_backtest_logging(log_dir: Path) -> Path:
    """Configures logging for a backtest run and returns the log file path."""
    log_dir.mkdir(exist_ok=True)
    log_filename = datetime.datetime.now().strftime("backtest_run_%Y%m%d_%H%M%S.log")
    log_filepath = log_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate logs in interactive environments
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filepath)
    stream_handler = logging.StreamHandler(sys.stdout) # Log to console as well
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    for handler in [file_handler, stream_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logging.info(f"Logging initialized. Log file: {log_filepath}")
    return log_filepath

def load_price_data(price_file_path: Path) -> Optional[pd.DataFrame]:
    """Loads historical price data and prepares it for backtesting."""
    logging.info(f"Loading historical price data from: {price_file_path}")
    if not price_file_path.exists():
        logging.error(f"Price data file not found: {price_file_path}")
        return None
    try:
        df_prices = pd.read_parquet(price_file_path)
        if not isinstance(df_prices.index, pd.DatetimeIndex):
            df_prices.index = pd.to_datetime(df_prices.index)
        if not df_prices.index.is_monotonic_increasing:
            df_prices = df_prices.sort_index()
        logging.info(f"Successfully loaded and prepared price data. Shape: {df_prices.shape}")
        return df_prices
    except Exception as e:
        logging.error(f"Failed to load or process price data: {e}", exc_info=True)
        return None

def find_and_pair_selection_files(selection_dir: Path) -> List[Tuple[Path, Path]]:
    """Finds and pairs selection data (.parquet) and parameter (.json) files."""
    logging.info(f"Searching for selection files in: {selection_dir}")
    if not selection_dir.is_dir():
        logging.error(f"Selection directory not found: {selection_dir}")
        return []
    
    all_json_files = list(selection_dir.glob("20*.json"))
    all_parquet_files = list(selection_dir.glob("20*.parquet"))
    
    param_map = {extract_date_from_string(f.name): f for f in all_json_files}
    
    file_pairs = []
    for data_file in sorted(all_parquet_files):
        date_key = extract_date_from_string(data_file.name)
        if date_key in param_map:
            file_pairs.append((data_file, param_map[date_key]))
        else:
            logging.warning(f"No matching parameter file for data file: {data_file.name}")
            
    logging.info(f"Found {len(file_pairs)} paired data and parameter files.")
    return file_pairs

# --- Core Backtesting Logic ---

def run_single_backtest(
    selection_date: str,
    scheme_name: str,
    ticker_weights: Dict[str, float],
    df_adj_close: pd.DataFrame,
    risk_free_rate_daily: float,
    ) -> Optional[Dict[str, Any]]:
    """
    Runs a simple T+1 buy to T+2 sell backtest for a given portfolio.
    This is the complete, unmodified core logic from the original notebook.
    """
    logging.info("-" * 30)
    logging.info(f"Initiating Backtest Run...")
    logging.info(f"  Date          : {selection_date}")
    logging.info(f"  Scheme        : {scheme_name}")
    logging.info(f"  Num Tickers   : {len(ticker_weights)}")

    try:
        df_prices = df_adj_close
        all_trading_dates = df_prices.index
        selection_timestamp = pd.Timestamp(selection_date)
        
        actual_selection_date_used = selection_timestamp
        try:
            selection_idx = all_trading_dates.get_loc(selection_timestamp)
        except KeyError:
            # If exact date not found, find the closest previous trading day (forward fill)
            indexer = all_trading_dates.get_indexer([selection_timestamp], method='ffill')
            if indexer[0] == -1: # Date is before the first date in the index
                logging.error(f"  Error: Selection date {selection_date} is before any available price data.")
                return None
            
            selection_idx = indexer[0]
            actual_selection_date_used = all_trading_dates[selection_idx]
            logging.warning(f"  Warning: Exact selection date {selection_date} not found. Using previous available date: {actual_selection_date_used.strftime('%Y-%m-%d')}")

        if selection_idx + 1 >= len(all_trading_dates):
            logging.error(f"  Error: No trading date found after selection date ({actual_selection_date_used.strftime('%Y-%m-%d')}). Cannot determine buy date.")
            return None
        buy_date = all_trading_dates[selection_idx + 1]

        if selection_idx + 2 >= len(all_trading_dates):
            logging.error(f"  Error: No trading date found after buy date ({buy_date.strftime('%Y-%m-%d')}). Cannot determine sell date.")
            return None
        sell_date = all_trading_dates[selection_idx + 2]

        logging.info(f"  Selection Date Used: {actual_selection_date_used.strftime('%Y-%m-%d')}")
        logging.info(f"  Buy Date           : {buy_date.strftime('%Y-%m-%d')}")
        logging.info(f"  Sell Date          : {sell_date.strftime('%Y-%m-%d')}")

        returns = []
        portfolio_return = 0.0
        total_weight_traded = 0.0
        
        relevant_tickers = [t for t in ticker_weights.keys() if t in df_prices.columns]
        price_subset = df_prices.loc[[buy_date, sell_date], relevant_tickers]

        num_successful_trades = 0
        for ticker, weight in ticker_weights.items():
            if ticker not in price_subset.columns:
                logging.warning(f"    - Ticker {ticker} not in price data. Skipping.")
                continue

            buy_price = price_subset.at[buy_date, ticker]
            sell_price = price_subset.at[sell_date, ticker]

            if pd.isna(buy_price) or buy_price <= 0 or pd.isna(sell_price):
                logging.warning(f"    - Invalid price data for {ticker} (Buy: {buy_price}, Sell: {sell_price}). Skipping trade.")
                continue
            
            trade_return = (sell_price - buy_price) / buy_price
            returns.append(trade_return)
            portfolio_return += trade_return * weight
            total_weight_traded += weight
            num_successful_trades += 1

        # --- Calculate Performance Metrics ---
        metrics = {
            'portfolio_return': portfolio_return,
            'portfolio_return_normalized': portfolio_return / total_weight_traded if abs(total_weight_traded) > 1e-9 else 0.0,
            'num_selected_tickers': len(ticker_weights),
            'num_attempted_trades': len(relevant_tickers),
            'num_successful_trades': num_successful_trades,
            'num_failed_or_skipped_trades': len(ticker_weights) - num_successful_trades,
            'total_weight_traded': total_weight_traded,
            'win_rate': np.nan, 'average_return': np.nan, 'std_dev_return': np.nan, 'sharpe_ratio_period': np.nan,
        }

        if num_successful_trades > 0:
            returns_array = np.array(returns)
            metrics['win_rate'] = np.sum(returns_array > 0) / num_successful_trades
            metrics['average_return'] = np.mean(returns_array)
            metrics['std_dev_return'] = np.std(returns_array, ddof=1) if num_successful_trades > 1 else 0.0
            
            excess_return = metrics['average_return'] - risk_free_rate_daily
            std_dev = metrics['std_dev_return']
            if std_dev > 1e-9:
                metrics['sharpe_ratio_period'] = excess_return / std_dev
            else:
                metrics['sharpe_ratio_period'] = np.inf * np.sign(excess_return) if abs(excess_return) > 1e-9 else 0.0
            
            logging.info(f"  Trades Executed: {num_successful_trades}/{len(ticker_weights)}")
            logging.info(f"  Portfolio Return : {metrics['portfolio_return']:.4f}")
            logging.info(f"  Win Rate         : {metrics['win_rate']:.2%}")
        else:
            logging.warning(f"  No successful trades executed out of {len(ticker_weights)} attempted.")

        backtest_results = {
            "run_inputs": {
                "selection_date": selection_date,
                "actual_selection_date_used": actual_selection_date_used.strftime('%Y-%m-%d'),
                "scheme_name": scheme_name,
                "buy_date": buy_date.strftime('%Y-%m-%d'),
                "sell_date": sell_date.strftime('%Y-%m-%d'),
            },
            "metrics": metrics,
        }
        logging.info(f"Backtest simulation for '{scheme_name}' on {selection_date} completed.")
        return backtest_results

    except Exception as e:
        logging.critical(f"  FATAL ERROR during backtest run for {selection_date}, {scheme_name}: {e}", exc_info=True)
        return None

def update_and_save_results(
    new_records: List[Dict[str, Any]],
    results_path: Path,
    unique_key_cols: List[str]
    ):
    """Loads existing results, merges new records, deduplicates, and saves."""
    logging.info(f"Updating results file at: {results_path}")
    
    if not new_records:
        logging.info("No new records to add. No changes made.")
        return

    df_new = pd.DataFrame(new_records)

    if results_path.exists():
        logging.info(f"Loading {results_path.name} to merge with new results.")
        df_old = pd.read_parquet(results_path)
        # Align columns before concat to prevent schema issues
        all_cols = sorted(list(set(df_old.columns) | set(df_new.columns)))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        logging.info("No existing results file found. Creating a new one.")
        df_combined = df_new

    # Deduplicate, keeping the latest run based on run_timestamp
    df_combined['run_timestamp'] = pd.to_datetime(df_combined['run_timestamp'])
    
    # Fill NaN for key columns before drop_duplicates to ensure they are treated correctly
    for key in unique_key_cols:
        if key not in df_combined.columns:
            logging.warning(f"Unique key '{key}' not in DataFrame. Skipping deduplication on this key.")
            continue
    
    # Final sort for consistent output
    df_final = df_combined.sort_values('run_timestamp').drop_duplicates(subset=unique_key_cols, keep='last')
    df_final = df_final.sort_values(by=['selection_date', 'scheme'], ascending=[False, True])
    
    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(results_path, index=False)
    # Optionally save to CSV as well for easy viewing
    df_final.to_csv(results_path.with_suffix('.csv'), index=False, float_format='%.6f')
    
    logging.info(f"Successfully saved {len(df_final)} consolidated records to {results_path.name}")


# IN src/utils.py, REPLACE THE OLD process_backtest_for_pair FUNCTION WITH THIS ONE:

def process_backtest_for_pair(
    data_file: Path,
    param_file: Path,
    df_adj_close: pd.DataFrame,
    risk_free_rate_daily: float,
    run_timestamp: str,
    log_filepath: Path
    ) -> List[Dict[str, Any]]:
    """High-level wrapper to process a single pair of data/param files."""
    records = []
    try:
        selection_date = extract_date_from_string(data_file.name)
        with open(param_file, 'r') as f:
            params = json.load(f) # params is a flat dictionary with correct key names
        
        df_selection = pd.read_parquet(data_file)
        weight_cols = [c for c in df_selection.columns if c.startswith('Weight_')]
        
        for weight_col in weight_cols:
            scheme_name = weight_col.split('_')[-1]
            ticker_weights = df_selection[weight_col].dropna().to_dict()
            
            if not ticker_weights:
                logging.warning(f"No tickers for scheme '{scheme_name}' on {selection_date}. Skipping.")
                continue

            result = run_single_backtest(selection_date, scheme_name, ticker_weights, df_adj_close, risk_free_rate_daily)
            
            if result:
                # --- CORRECTED LOGIC ---
                # Since the keys in the JSON file already match the desired column names,
                # we can directly unpack the `params` dictionary into the record.
                
                record = {
                    'run_timestamp': run_timestamp,
                    'log_file': log_filepath.name,
                    'selection_date': selection_date,
                    'actual_selection_date_used': result.get('run_inputs', {}).get('actual_selection_date_used'),
                    'scheme': scheme_name,
                    **params,  # <-- This correctly unpacks all key-value pairs from the JSON
                    **result['metrics'] # This unpacks all performance metrics
                }
                
                records.append(record)

    except Exception as e:
        logging.error(f"Failed to process pair {data_file.name}/{param_file.name}: {e}", exc_info=True)

    return records



# =================================================================================
# NEW PLOTTING FUNCTION FOR CUMULATIVE RETURNS - APPEND TO src/utils.py
# =================================================================================

def plot_cumulative_returns(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    scheme_col: str,
    strategy_id_cols: list,
    top_n_strategies: int = 1
    ):
    """
    Plots the cumulative return equity curve for the best performing strategies.
    
    Args:
        df (pd.DataFrame): The master results DataFrame.
        date_col (str): The name of the date column.
        return_col (str): The name of the daily return column.
        scheme_col (str): The name of the scheme column.
        strategy_id_cols (list): List of column names that uniquely identify a strategy.
        top_n_strategies (int): The number of top strategies to plot based on final return.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")

    # Identify the top N strategies based on their final cumulative return
    grouped = df.groupby(strategy_id_cols + [scheme_col])
    final_returns = grouped[return_col].apply(lambda x: (1 + x).prod() - 1)
    top_strategies = final_returns.nlargest(top_n_strategies).index
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    for strategy in top_strategies:
        # Unpack the multi-index tuple
        strategy_params = strategy[:-1]
        scheme = strategy[-1]
        
        # Filter the DataFrame for this specific strategy and scheme
        mask = (df[strategy_id_cols] == strategy_params).all(axis=1) & (df[scheme_col] == scheme)
        subset = df[mask].sort_values(by=date_col)

        if not subset.empty:
            # Calculate cumulative return (equity curve)
            equity_curve = (1 + subset[return_col]).cumprod()
            
            # Create a clean label for the legend
            params_str = ', '.join([f"{col.split('_')[-1]}={val}" for col, val in zip(strategy_id_cols, strategy_params)])
            label = f"Scheme: {scheme} ({params_str})"
            
            ax.plot(subset[date_col], equity_curve, label=label, linewidth=2)

    ax.set_title(f'Equity Curve for Top {top_n_strategies} Strategy Run(s)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (1 = breakeven)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle=':', linewidth=0.6)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.75) # Breakeven line
    fig.autofmt_xdate()
    plt.show()



# =================================================================================
# NEW ADDITIONS FOR ANALYSIS AND VISUALIZATION - APPEND TO src/utils.py
# =================================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.dates import DateFormatter

def plot_evolving_annualized_sharpe(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    scheme_col: str,
    annual_risk_free_rate: float,
    trading_days_per_year: int = 252,
    min_periods_for_sharpe: int = 10
    ) -> pd.DataFrame:
    """
    Calculates and plots the evolving annualized Sharpe Ratio for different schemes.

    The function processes a DataFrame of daily returns, calculates the expanding
    mean and standard deviation of these returns for each scheme, and then
    computes the daily and annualized Sharpe Ratios. The results are plotted
    over time, showing how the risk-adjusted performance of each scheme evolves.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in [date_col, return_col, scheme_col]):
        raise ValueError(f"DataFrame must contain columns: {date_col}, {return_col}, {scheme_col}")
    if not isinstance(trading_days_per_year, int) or trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be a positive integer.")
    if not isinstance(min_periods_for_sharpe, int) or min_periods_for_sharpe < 2:
        raise ValueError("min_periods_for_sharpe must be an integer greater than or equal to 2.")

    # --- 1. Data Preparation ---
    df_analysis = df.copy()
    df_analysis[date_col] = pd.to_datetime(df_analysis[date_col]).dt.normalize()
    df_analysis = df_analysis.dropna(subset=[date_col])
    df_analysis = df_analysis.sort_values(by=[scheme_col, date_col])

    # --- 2. Calculate Daily Risk-Free Rate ---
    daily_risk_free_rate = annual_risk_free_rate / trading_days_per_year

    # --- 3. Calculate Evolving Metrics and Sharpe Ratio ---
    results_list = []
    for scheme_name, group in df_analysis.groupby(scheme_col):
        group['expanding_mean_return'] = group[return_col].expanding(min_periods=min_periods_for_sharpe).mean()
        group['expanding_std_return'] = group[return_col].expanding(min_periods=min_periods_for_sharpe).std()

        numerator = group['expanding_mean_return'] - daily_risk_free_rate
        denominator = group['expanding_std_return']

        group['daily_sharpe_ratio'] = np.where(
            (denominator > 1e-9) & (denominator.notna()),
            numerator / denominator,
            np.nan
        )
        group['annualized_sharpe_ratio'] = group['daily_sharpe_ratio'] * np.sqrt(trading_days_per_year)
        results_list.append(group)

    if not results_list:
        print("Warning: No data after processing. Cannot generate plot.")
        return pd.DataFrame()
        
    df_results = pd.concat(results_list)

    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    all_plotted_dates_objects = []

    for scheme in df_results[scheme_col].unique():
        subset_plot = df_results[df_results[scheme_col] == scheme].dropna(subset=['annualized_sharpe_ratio', date_col])
        if not subset_plot.empty:
            ax.plot(subset_plot[date_col], subset_plot['annualized_sharpe_ratio'],
                    label=f"Scheme: {scheme}", linewidth=2, marker='o', markersize=4)
            all_plotted_dates_objects.extend(subset_plot[date_col].tolist())

    ax.set_title(f'Evolving Annualized Sharpe Ratio (vs. {annual_risk_free_rate*100:.1f}% Ann. Risk-Free Rate)', fontsize=16)
    ax.set_xlabel(f'Date ({date_col})', fontsize=12)
    ax.set_ylabel(f'Annualized Sharpe Ratio (using {trading_days_per_year} days/year)', fontsize=12)

    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=10)
    
    ax.grid(True, which='both', linestyle=':', linewidth=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.75)

    if all_plotted_dates_objects:
        unique_plotted_dates = sorted(list(pd.Series(all_plotted_dates_objects).unique()))
        unique_plotted_dates = [pd.Timestamp(d) for d in unique_plotted_dates if pd.notna(d)]
        if unique_plotted_dates:
            ax.set_xticks(unique_plotted_dates)
            ax.xaxis.set_major_formatter(DateFormatter("%m/%d/%y"))
            fig.autofmt_xdate(rotation=30, ha='right')

    plt.show()
    return df_results



# =================================================================================
# NEW ADDITIONS FOR PIPELINE ORCHESTRATION - APPEND TO src/utils.py
# =================================================================================

import re
from typing import List, Optional

# Note: Assumes a function `get_recent_files_in_directory` already exists.
# If not, it would be defined here. We will assume it's refactored
# to accept a Path object directly, e.g., `directory_path: Path`.

def extract_and_sort_dates_from_filenames(filenames: List[str]) -> List[str]:
    """Extracts unique date strings (YYYY-MM-DD) from filenames and sorts them."""
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    all_dates = {match.group(0) for filename in filenames if (match := date_pattern.search(filename))}
    return sorted(list(all_dates))

def print_list_in_columns(items: List[str], num_columns: int = 5):
    """Prints a list of strings in a numbered, multi-column format for readability."""
    if not items:
        print("No items to display.")
        return
    for i in range(0, len(items), num_columns):
        row_slice = items[i : i + num_columns]
        row_items = [f"  {i+j:<4} {item}" for j, item in enumerate(row_slice)]
        print("  ".join(row_items))

def parse_str_to_slice(slice_str: str) -> Optional[slice]:
    """Parses a string like "start:stop:step" into a slice object."""
    try:
        parts = [int(p) if p.strip() else None for p in slice_str.split(':')]
        return slice(*parts)
    except (ValueError, IndexError):
        return None

def prompt_for_slice_update(variable_name: str, current_value: slice) -> slice:
    """Displays a slice's current value and prompts the user to keep or change it."""
    s = current_value
    current_value_str = f"{s.start or ''}:{s.stop or ''}:{s.step or ''}"
    while True:
        prompt = (
            f"\n-> The current {variable_name} is: '{current_value_str}'\n"
            f"   Enter a new slice (e.g., ':10', '-5:', '::-1') or press ENTER to continue: "
        )
        user_input = input(prompt).strip()
        if not user_input:
            print("   Continuing with the current value.")
            return current_value
        new_slice = parse_str_to_slice(user_input)
        if new_slice is not None:
            print(f"   {variable_name} updated.")
            return new_slice
        else:
            print(f"   Error: Invalid slice format '{user_input}'. Please try again.")

def create_pipeline_config_file(
    config_path: Path,
    date_str: str,
    downloads_dir: Path,
    dest_dir: Path,
    annual_risk_free_rate: float,
    trading_days_per_year: int
    ):
    """Creates a config.py file with dynamic paths and parameters."""
    daily_risk_free_rate = annual_risk_free_rate / trading_days_per_year
    
    # Use .as_posix() to ensure cross-platform compatible path strings
    config_content = f"""# config.py
# This file is auto-generated by py0. DO NOT EDIT MANUALLY.

from pathlib import Path

# --- File path configuration ---
DATE_STR = '{date_str}'
DOWNLOAD_DIR = Path('{downloads_dir.as_posix()}')
DEST_DIR = Path('{dest_dir.as_posix()}')

# --- Analysis Parameters ---
ANNUAL_RISK_FREE_RATE = {annual_risk_free_rate}
TRADING_DAYS_PER_YEAR = {trading_days_per_year}
DAILY_RISK_FREE_RATE = {daily_risk_free_rate}
"""
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"Successfully created config file: {config_path}")


# In src/utils.py, find and REPLACE the old get_recent_files_in_directory function

import os
from pathlib import Path
from typing import List

def get_recent_files_in_directory(
    directory_path: Path, 
    prefix: str, 
    extension: str, 
    count: int
    ) -> List[str]:
    """
    Finds and returns a list of the most recent filenames in a given directory
    that match a specific prefix and extension.

    Args:
        directory_path (Path): The pathlib.Path object for the directory to search.
        prefix (str): The required starting string of the filenames.
        extension (str): The required file extension (e.g., 'parquet', 'csv').
        count (int): The maximum number of recent files to return.

    Returns:
        List[str]: A list of filenames, sorted from most recent to oldest.
    """
    if not directory_path.is_dir():
        print(f"Warning: Directory not found at {directory_path}")
        return []

    # Construct the glob pattern to find matching files
    pattern = f"{prefix}*.{extension}"
    
    # Find all matching files and get their modification times
    files_with_mtime = []
    for f in directory_path.glob(pattern):
        try:
            # Use f.stat() to get file metadata, including modification time
            mtime = f.stat().st_mtime
            files_with_mtime.append((f.name, mtime))
        except FileNotFoundError:
            # This can happen in rare race conditions if a file is deleted
            # while the loop is running.
            continue
            
    # Sort the files by modification time in descending order (most recent first)
    files_with_mtime.sort(key=lambda x: x[1], reverse=True)
    
    # Return only the filenames from the sorted list, up to the specified count
    sorted_filenames = [filename for filename, mtime in files_with_mtime]
    
    return sorted_filenames[:count]


