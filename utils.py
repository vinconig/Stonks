import pandas as pd
from typing import Any
import numpy as np

def load_comp_data(df: pd.DataFrame, comp: str) -> pd.DataFrame:
    """
    Return a dataframe containing the first column of df (renamed to 'day')
    and all columns whose names start with the string `comp`.

    Raises:
        TypeError: if df is not a pandas DataFrame.
        ValueError: if df has no columns or no columns start with `comp`.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.columns.size == 0:
        raise ValueError("Input dataframe has no columns")

    #we dont want to train on absolute time so we will convert into enumeration
    
    
    matching = [c for c in df.columns if c.startswith(comp)]

    if len(matching) == 0:
        raise ValueError(f"There is no Company in the data set with that ticker symbol: {comp}")

    cols = [c for c in matching]
    result = df.loc[:, cols].copy()
    result.insert(0, "timestep", np.arange(len(result)))
    result = result.dropna(how="any")

    return result

def compute_features(df: pd.DataFrame, w: int, log_return: bool = True, tr: bool = True, rsd: bool = True, ntv: bool = True, cut: bool = True) -> Any:
    """
    Compute optional features on a price dataframe and add a column for the
    logarithmic next-day maximum return.

    If df has more than 7 columns, returns the exact string:
    "Doesnt have expected shape - make sure to run load_comp_data first" 

    Parameters:
        df: pandas DataFrame (expected to include close and high at minimum)
        w: window size for rolling calculations (positive integer)
        log_return, tr, rsd, ntv: booleans selecting which features to compute
        cut: if True, drop initial rows without full rolling window, only if features with rolling windows are calculated. Turn False if little data available.

    Returns:
        pd.DataFrame with added feature columns (or the string above on shape mismatch)
    """

    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(w, int) or w <= 0:
        raise ValueError("w must be a positive integer")
    if df.columns.size > 7:
        return "Doesn't have expected shape - make sure to run load_comp_data first!"

    # helper to find a column by substring (case-insensitive)
    cols_map = {c.lower(): c for c in df.columns}
    def find_col(sub: str):
        for lc, orig in cols_map.items():
            if sub in lc:
                return orig
        return None

    open_col = find_col("open")
    adj_col = find_col("adj")
    close_col = find_col("close")
    high_col = find_col("high")
    low_col = find_col("low")
    vol_col = find_col("vol") or find_col("volume")

    if open_col is None:
        raise ValueError("Open price column not found in dataframe")
    if close_col is None:
        raise ValueError("Close price column not found in dataframe")
    if high_col is None:
        raise ValueError("High price column not found in dataframe")
    if low_col is None:
        raise ValueError("Low price column not found in dataframe")
    if vol_col is None and ntv:
        raise ValueError("Volume column not found in dataframe but ntv requested")
    
        adj_col = close_col  # use close if no adjusted close available

    result = df.copy()

    #drop the adj col from the dataframe
    
    result = result.drop(columns=["adj_col"])
    # log returns (like exposé)
    need_log = log_return or rsd  # rsd requires log returns
    if need_log:
        result["log_return"] = np.log(result[close_col]/result[open_col])

    # True Range this is the better definition of true range i found, it deviates from our Exposé but its superior
    if tr:
        prev_close = result[close_col].shift(1)
        tr1 = result[high_col] - result[low_col]
        tr2 = (result[high_col] - prev_close).abs()
        tr3 = (result[low_col] - prev_close).abs()
        result["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Rolling standard deviation of log returns - slightly different from exposé, but more robust
    if rsd:
        result[f"rsd_{w}"] = result["log_return"].rolling(window=w, min_periods=1).std()

    # Normalized trading volume: volume / rolling mean(volume, w)
    if ntv:
        if vol_col is None:
            raise ValueError("Volume column not found in dataframe but ntv requested")
        rolling_vol_mean = result[vol_col].rolling(window=w, min_periods=1).mean()
        # avoid division by zero
        result[f"ntv_{w}"] = result[vol_col] / rolling_vol_mean.replace(0, np.nan)

    # Logarithmic next-day maximum return (use next day's high vs today's close)
    result["log_next_day_max_return"] = np.log(result[high_col].shift(-1) / result[close_col])

    # drop rows with any NaNs introduced by shifts/rolling 
    result = result.dropna(how="any").reset_index(drop=True)

    if rsd or ntv:
        if cut:
            # drop initial rows without full rolling window
            result = result.iloc[w-1:].reset_index(drop=True)
    

    return result
