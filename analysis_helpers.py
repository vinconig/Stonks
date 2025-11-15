#analyze helpersimport pandas as pd
import numpy as np
import pandas as pd

def extract_stocks(df: pd.DataFrame):
    """
    Extract stock tickers by splitting column names like 'MMM_open'.
    Assumes columns follow <TICKER>_<FIELD>.
    """
    tickers = set()
    for col in df.columns:
        if "_" in col:
            ticker = col.split("_")[0]
            tickers.add(ticker)
    return sorted(tickers)


def compute_volatility(df, ticker, window=20):
    """
    Compute rolling 20-day volatility based on close returns.
    """
    close_col = f"{ticker}_close"
    if close_col not in df:
        return np.nan
    
    returns = df[close_col].pct_change()
    vol = returns.rolling(window).std()
    
    return vol.mean()  # overall volatility metric for bucket assignment


def compute_liquidity(df, ticker, window=20):
    """
    Compute liquidity measured as average volume.
    """
    vol_col = f"{ticker}_volume"
    if vol_col not in df:
        return np.nan
    
    return df[vol_col].rolling(window).mean().mean()


def assign_bucket(value, quantiles):
    """
    Given a scalar and quantile breakpoints (q_low, q_high),
    return the appropriate bucket.
    """
    q_low, q_high = quantiles
    if pd.isna(value):
        return "missing"
    if value <= q_low:
        return "low"
    elif value >= q_high:
        return "high"
    else:
        return "medium"


def analyze_stocks(df):
    tickers = extract_stocks(df)

    vol_dict = {}
    liq_dict = {}

    # ---- compute metrics ----
    for t in tickers:
        vol_dict[t] = compute_volatility(df, t)
        liq_dict[t] = compute_liquidity(df, t)

    vol_series = pd.Series(vol_dict, name="volatility")
    liq_series = pd.Series(liq_dict, name="liquidity")

    # ---- define bucketing thresholds ----
    # low ≤ 30%ile, high ≥ 70%ile
    vol_q_low  = vol_series.quantile(0.30)
    vol_q_high = vol_series.quantile(0.70)

    liq_q_low  = liq_series.quantile(0.30)
    liq_q_high = liq_series.quantile(0.70)

    result = pd.DataFrame({
        "volatility": vol_series,
        "vol_bucket": vol_series.apply(assign_bucket, quantiles=(vol_q_low, vol_q_high)),
        "liquidity": liq_series,
        "liq_bucket": liq_series.apply(assign_bucket, quantiles=(liq_q_low, liq_q_high)),
    })

    return result.sort_values("volatility")