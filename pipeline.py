# pipeline.py
# Seasonality + fundamentals + scoring + reliability label
# Corrected and robust implementation for the seasonal/intraday app.
#
# - Uses yf.Ticker for history and fundamentals
# - Monthly resampling uses 'ME' (month end)
# - Identifies buy months (lowest avg_ret) and peak months (highest avg_ret)
# - Computes seasonality score based on the PEAK month (strength of upward seasonality)
# - Computes binomial p-value and reliability label
# - analyze_universe_with_fundamentals returns only candidates that pass score & quality filters

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math

YEARS_DEFAULT = 10

# ------------------ Price history & seasonality ------------------
def fetch_price_history(ticker, years=YEARS_DEFAULT):
    """
    Fetch daily price history for `years` years. Returns DataFrame or None.
    """
    try:
        period = f"{years}y"
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval="1d", actions=False)
        if hist is None or hist.empty:
            return None
        # keep standard columns
        cols = ['Open','High','Low','Close','Volume']
        for c in cols:
            if c not in hist.columns:
                hist[c] = np.nan
        hist = hist[cols].copy()
        hist.index = pd.to_datetime(hist.index)
        return hist
    except Exception:
        return None

def resample_monthly(df):
    """
    Convert daily df to monthly rows with 'open' (first), 'close' (last), 'ret' (open->close), 'volume' (sum), year, month.
    Uses 'ME' (month end) resampling to avoid FutureWarning.
    """
    monthly_close = df['Close'].resample('ME').last()
    monthly_open = df['Open'].resample('ME').first()
    monthly_vol = df['Volume'].resample('ME').sum()
    # guard against zeros/NaN
    monthly_ret = (monthly_close - monthly_open) / monthly_open
    md = pd.DataFrame({'open': monthly_open, 'close': monthly_close, 'ret': monthly_ret, 'volume': monthly_vol})
    md = md.dropna(subset=['open','close','ret'])
    if md.empty:
        return md
    md['year'] = md.index.year
    md['month'] = md.index.month
    return md

def compute_monthly_stats(df):
    """
    df: daily history DataFrame
    returns: DataFrame with 12 rows (month 1..12) and stats: avg_ret, median_ret, pct_positive, std, avg_vol, years_count
    """
    md = resample_monthly(df)
    stats = []
    for m in range(1,13):
        mdf = md[md['month'] == m]
        if mdf.empty:
            stats.append({'month': m, 'avg_ret': np.nan, 'median_ret': np.nan, 'pct_positive': np.nan, 'std': np.nan, 'avg_vol': np.nan, 'years_count': 0})
            continue
        avg_ret = float(mdf['ret'].mean())
        median_ret = float(mdf['ret'].median())
        pct_positive = float((mdf['ret'] > 0).sum() / len(mdf) * 100)
        std = float(mdf['ret'].std())
        vol_mean = float(mdf['volume'].mean())
        stats.append({'month': m, 'avg_ret': avg_ret, 'median_ret': median_ret, 'pct_positive': pct_positive, 'std': std, 'avg_vol': vol_mean, 'years_count': len(mdf)})
    sdf = pd.DataFrame(stats)
    return sdf

def build_seasonality_matrix(df):
    """
    Build a years x months matrix of monthly returns for heatmap display.
    Returns matrix DataFrame where rows are years and columns are months (1..12).
    """
    md = resample_monthly(df)
    if md.empty:
        return pd.DataFrame()
    pivot = md.pivot_table(index='year', columns='month', values='ret')
    pivot = pivot.reindex(columns=range(1,13))
    return pivot

# ------------------ Fundamentals (basic via yfinance) ------------------
def fetch_fundamentals_yf(ticker):
    """
    Attempt to gather basic fundamentals from yfinance.info
    Not exhaustive; if values missing they will be None.
    """
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
    except Exception:
        info = {}

    fund = {}
    fund['marketCap'] = info.get('marketCap')
    fund['sector'] = info.get('sector')
    fund['longName'] = info.get('longName')
    fund['averageVolume'] = info.get('averageVolume') or info.get('averageVolume10d') or info.get('averageDailyVolume10Day')
    fund['debtToEquity'] = info.get('debtToEquity') or info.get('debtToEquityRatio') or info.get('totalDebt')
    fund['totalRevenue'] = info.get('totalRevenue') or info.get('revenue') or info.get('revenueGrowth')
    fund['currency'] = info.get('currency')
    return fund

# ------------------ Scoring & quality filters ------------------
def seasonality_strength_score_improved(row):
    """
    Improved 0-100 score using the month's statistics row (expected to be a peak month row).
    Components:
      - pct_positive (40%)
      - avg_ret scaled (30%)
      - std (10%, lower is better)
      - avg_vol scaling (10%)
      - years_count reliability (10%)
    """
    if row is None or not isinstance(row, dict):
        return 0.0

    pct_pos = float(row.get('pct_positive', 0) or 0)
    avg_ret = float(row.get('avg_ret', 0) or 0)
    std = float(row.get('std', 0) or 0)
    vol = float(row.get('avg_vol', 0) or 0)
    years = int(row.get('years_count', 0) or 0)

    # avg_ret scale (-0.3..0.3 -> 0..100)
    ar = max(min((avg_ret + 0.3) / 0.6 * 100, 100), 0)

    # std scaling: assume 0..0.5 range (lower std => higher score)
    sd = max(min((0.5 - std) / 0.5 * 100, 100), 0)

    # volume scaling: use log1p to avoid huge numbers
    vol_score = 0
    if vol > 0:
        vol_score = min(np.log10(vol + 1) / 6 * 100, 100)

    years_score = min(years / YEARS_DEFAULT * 100, 100)

    score = 0.40 * pct_pos + 0.30 * ar + 0.10 * sd + 0.10 * vol_score + 0.10 * years_score
    return float(max(min(score, 100), 0))

def quality_filters_ok(fund, thresholds):
    """
    thresholds: dict with keys marketcap_min, min_avg_volume, max_debt_to_equity
    Returns True only if fundamentals pass thresholds (missing data is treated conservatively as fail).
    """
    # Market cap check
    mc = fund.get('marketCap')
    if mc is None:
        return False
    try:
        if mc < thresholds.get('marketcap_min', 0):
            return False
    except Exception:
        return False

    # average volume
    avg_vol = fund.get('averageVolume')
    if avg_vol is None:
        return False
    try:
        if avg_vol < thresholds.get('min_avg_volume', 0):
            return False
    except Exception:
        return False

    # debt-to-equity (if available)
    d2e = fund.get('debtToEquity')
    if d2e is not None:
        try:
            d = float(d2e)
            if d > thresholds.get('max_debt_to_equity', 10):
                return False
        except Exception:
            return False

    return True

# ------------------ Binomial p-value for month success ------------------
def binomial_p_value(k, n, p0=0.5):
    """
    Return P(X >= k) for X ~ Binomial(n, p0)
    Uses direct combination sums. Works for small n (~10).
    """
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    s = 0.0
    for i in range(k, n+1):
        comb = math.comb(n, i)
        s += comb * (p0**i) * ((1-p0)**(n-i))
    return s

def reliability_label(score, pct_positive, years_count, binom_p, thresholds):
    """
    Return 'High', 'Medium', or 'Low' reliability based on thresholds.
    """
    if (score >= thresholds.get('score_high', 80) and
        pct_positive >= thresholds.get('pctpos_high', 80) and
        years_count >= thresholds.get('years_high', 8) and
        binom_p <= thresholds.get('p_high', 0.05)):
        return "High"
    if (score >= thresholds.get('score_med', 65) and
        pct_positive >= thresholds.get('pctpos_med', 70) and
        years_count >= thresholds.get('years_med', 6) and
        binom_p <= thresholds.get('p_med', 0.10)):
        return "Medium"
    return "Low"

# ------------------ Full analysis for a single ticker ------------------
def analyze_ticker_full(ticker, years=YEARS_DEFAULT):
    """
    Full pipeline for a ticker: fetch history -> monthly stats -> buy/peak months -> score -> heatmap matrix
    """
    hist = fetch_price_history(ticker, years=years)
    if hist is None or hist.empty:
        return None

    monthly_stats = compute_monthly_stats(hist)
    if monthly_stats is None or monthly_stats.empty:
        return None

    # Determine buy (lowest avg_ret months) and peak (highest avg_ret months)
    valid = monthly_stats.dropna(subset=['avg_ret'])
    if valid.empty:
        return None

    # buy months: months with lowest average return (suggested months to buy)
    buy_months = valid.sort_values('avg_ret').head(2)['month'].astype(int).tolist()

    # peak months: months with highest average return (suggested months to sell)
    peak_months = valid.sort_values('avg_ret', ascending=False).head(2)['month'].astype(int).tolist()

    # Choose the PEAK month row (highest avg_ret) to compute seasonality strength
    try:
        peak_row = valid.sort_values('avg_ret', ascending=False).iloc[0].to_dict()
    except Exception:
        peak_row = None

    score = seasonality_strength_score_improved(peak_row) if peak_row is not None else 0.0
    matrix = build_seasonality_matrix(hist)

    result = {
        'ticker': ticker,
        'monthly_stats': monthly_stats,
        'best_month_row': peak_row,
        'buy_months': buy_months,
        'peak_months': peak_months,
        'score': score,
        'heatmap_matrix': matrix
    }
    return result

def analyze_universe_with_fundamentals(tickers, years=YEARS_DEFAULT, thresholds=None, min_score=50):
    """
    Analyze a list of tickers and return candidates that pass both seasonality score and fundamental quality filters.
    Returns list of candidate dicts with fundamentals and labels.
    """
    if thresholds is None:
        thresholds = {
            'marketcap_min': 20000 * 1e7,   # ₹20,000 Cr
            'min_avg_volume': 50_000,
            'max_debt_to_equity': 2.0,
            'score_high': 80, 'pctpos_high': 80, 'years_high': 8, 'p_high': 0.05,
            'score_med': 65, 'pctpos_med': 70, 'years_med': 6, 'p_med': 0.10
        }

    candidates = []
    for t in tickers:
        try:
            res = analyze_ticker_full(t, years=years)
            if res is None:
                continue

            fund = fetch_fundamentals_yf(t)

            # Compute binomial p-value for the peak month success frequency
            peak_row = res.get('best_month_row')
            if peak_row is None:
                continue
            pct_pos = peak_row.get('pct_positive', 0) or 0
            years_count = int(peak_row.get('years_count', 0) or 0)
            k = int(round(pct_pos / 100 * years_count)) if years_count > 0 else 0
            binom_p = binomial_p_value(k, years_count, p0=0.5) if years_count > 0 else 1.0

            # Quality filters
            qok = quality_filters_ok(fund, thresholds)

            # Reliability label
            label = reliability_label(res['score'], pct_pos, years_count, binom_p, thresholds)

            c = {
                'ticker': t,
                'score': res['score'],
                'buy_months': res['buy_months'],
                'peak_months': res['peak_months'],
                'monthly_stats': res['monthly_stats'],
                'best_month_row': res['best_month_row'],
                'heatmap_matrix': res['heatmap_matrix'],
                'fundamentals': fund,
                'quality_pass': qok,
                'binom_p': binom_p,
                'reliability_label': label
            }

            if c['score'] >= min_score and c['quality_pass']:
                candidates.append(c)
        except Exception:
            # ignore ticker on any unexpected error to keep pipeline robust
            continue

    candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
    return candidates_sorted
