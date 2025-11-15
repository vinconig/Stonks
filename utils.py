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
    
    # keep rows starting from the first row that has no NaNs in any column
    mask = result.notna().all(axis=1)
    if not mask.any():
        raise ValueError("No row without NaNs found; cannot align dataset")
    first_idx = mask[mask].index[0]
    result = result.loc[first_idx:].reset_index(drop=True)

    # forward fill any remaining NaN values (should be rare after above step)
    result.fillna(method='ffill', inplace=True)

    return result



def compute_features(
    df: pd.DataFrame,
    w: int,
    cut: bool = True
) -> Any:
    """
    Compute optional features for OHLCV dataframe and add a column for the logarithmic next-day maximum return. 
    If df has more than 7 columns, returns the exact string: "Doesnt have expected shape - make sure to run load_comp_data first" 
    Parameters: 
    df: pandas DataFrame (expected to include close and high at minimum) 
    w: window size for rolling calculations (positive integer)
    cut: if True, removes first w-1 rows to account for rolling calculations
    among others:
        - volatility features
        - moving averages (SMA, EMA, TEMA, WMA)
        - Bollinger Bands
        - momentum/ROC
        - RSI
        - OBV
        - candle geometry features

    Returned columns are rolling z-normalized (except timestep + target).
    """

    # ---------------------------
    # Input Checks
    # ---------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(w, int) or w <= 0:
        raise ValueError("w must be a positive integer")
    if df.columns.size > 7:
        return "Doesn't have expected shape - make sure to run load_comp_data first!"

    # ---------------------------
    # Column detection helper
    # ---------------------------
    cols_map = {c.lower(): c for c in df.columns}

    def find_col(sub: str):
        for lc, orig in cols_map.items():
            if sub in lc:
                return orig
        return None

    open_col = find_col("open")
    adj_col = find_col("adj_close")
    close_col = find_col("close")
    high_col = find_col("high")
    low_col = find_col("low")
    vol_col = find_col("vol")

    if open_col is None: raise ValueError("Open column not found")
    if close_col is None: raise ValueError("Close column not found")
    if high_col is None: raise ValueError("High column not found")
    if low_col is None: raise ValueError("Low column not found")
    if vol_col is None: raise ValueError("Volume column not found")
    if adj_col is None: raise ValueError("Adjusted close column not found")

    # ---------------------------
    # Base copy + drop adj col
    # ---------------------------
    result = df.copy()
    result = result.drop(columns=[adj_col])

    # ======================================================
    #   ORIGINAL FEATURES (kept exactly as you wrote them)
    # ======================================================

    # log returns
    result["log_return"] = np.log(result[close_col] / result[open_col])

    #true range
    prev_close = result[close_col].shift(1)
    tr1 = result[high_col] - result[low_col]
    tr2 = (result[high_col] - prev_close).abs()
    tr3 = (result[low_col] - prev_close).abs()
    result["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # rolling std of log returns
    result[f"rsd_{w}"] = result["log_return"].rolling(window=w, min_periods=1).std()

    # normalized trading volume
    rolling_vol_mean = result[vol_col].rolling(window=w, min_periods=1).mean()
    result[f"ntv_{w}"] = result[vol_col] / rolling_vol_mean.replace(0, np.nan)

    # next-day max return target
    result["log_next_day_max_return"] = np.log(result[high_col].shift(-1) / result[close_col])


    #hl range
    result["hl_range"] = result[high_col] - result[low_col]

    # Parkinson volatility
    result["parkinson_vol"] = (
        (np.log(result[high_col] / result[low_col])) ** 2 / (4 * np.log(2))
    ).rolling(w, min_periods=1).mean()

    # Garman-Klass volatility
    result["garman_klass_vol"] = (
        0.5 * (np.log(result[high_col] / result[low_col])) ** 2 -
        (2 * np.log(2) - 1) * (np.log(result[close_col] / result[open_col])) ** 2
    ).rolling(w, min_periods=1).mean()

    # Rogers-Satchell volatility
    result["rogers_satchell_vol"] = (
        np.log(result[high_col] / result[close_col]) *
        np.log(result[high_col] / result[open_col]) +
        np.log(result[low_col] / result[close_col]) *
        np.log(result[low_col] / result[open_col])
    ).rolling(w, min_periods=1).mean()

    # ---------------- MOVING AVERAGES --------------------
    result["sma_w"] = result[close_col].rolling(w, min_periods=1).mean()
    result["ema_w"] = result[close_col].ewm(span=w, adjust=False).mean()
    result["wma_w"] = (
        result[close_col].rolling(w, min_periods=1)
        .apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)))
    )
    ema1 = result[close_col].ewm(span=w, adjust=False).mean()
    ema2 = ema1.ewm(span=w, adjust=False).mean()
    ema3 = ema2.ewm(span=w, adjust=False).mean()
    result["tema_w"] = 3 * ema1 - 3 * ema2 + ema3

    # ---------------- MOMENTUM ----------------------------
    result["momentum_w"] = result[close_col] - result[close_col].shift(w)
    result["roc_w"] = result[close_col].pct_change(w)

    # ---------------- BOLLINGER BANDS ---------------------
    mid = result["sma_w"]
    std = result[close_col].rolling(w, min_periods=1).std()

    result["bollinger_mid"] = mid
    result["bollinger_upper"] = mid + 2 * std
    result["bollinger_lower"] = mid - 2 * std
    result["bollinger_width"] = (result["bollinger_upper"] - result["bollinger_lower"]) / mid.replace(0, np.nan)
    result["bollinger_percent_b"] = (result[close_col] - result["bollinger_lower"]) / (
        result["bollinger_upper"] - result["bollinger_lower"]
    ).replace(0, np.nan)

    # ---------------- RSI --------------------------------
    delta = result[close_col].diff()
    up = delta.clip(lower=0).rolling(w).mean()
    down = (-delta.clip(upper=0)).rolling(w).mean()
    rs = up / down.replace(0, np.nan)
    result["rsi_w"] = 100 - (100 / (1 + rs))

    # ---------------- VOLUME FEATURES ---------------------
    # OBV
    sign = np.sign(result[close_col].diff()).replace(0, 0)
    result["obv"] = (sign * result[vol_col]).fillna(0).cumsum()

    # raw volume z-score (before global normalization)
    vol_mean = result[vol_col].rolling(w, min_periods=1).mean()
    vol_std = result[vol_col].rolling(w, min_periods=1).std()
    result["volume_z"] = (result[vol_col] - vol_mean) / vol_std.replace(0, np.nan)

    # ---------------- CANDLE GEOMETRY ---------------------
    result["candle_body"] = result[close_col] - result[open_col]
    result["candle_range_ratio"] = (
        result["candle_body"] / (result[high_col] - result[low_col]).replace(0, np.nan)
    )

    
    # ======================================================
    # ROLLING Z-SCORE NORMALIZATION (all features except target)
    # ======================================================
    for col in result.columns:
        if col not in ["timestep", "log_next_day_max_return"]:
            mean = result[col].rolling(window=w, min_periods=1).mean()
            std = result[col].rolling(window=w, min_periods=1).std()
            result[col] = (result[col] - mean) / std.replace(0, np.nan)
    
    if cut:
        result = result.iloc[w - 1:].reset_index(drop=True)

    result = result.dropna(how="any").reset_index(drop=True)
    return result
