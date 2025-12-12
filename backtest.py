# backtest.py
# Seasonal backtesting utilities for the seasonal_intraday_app
# - seasonal_backtest: simulates buying at first trading day of buy_month and selling at last trading day of sell_month
# - portfolio_backtest_from_alloc: simple portfolio-level backtest using allocation results

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Tuple, Dict

def _get_month_first_trade(df: pd.DataFrame, year: int, month: int):
    # df indexed by Timestamp (trading days). Return first trading day's row for given year,month
    try:
        sub = df[(df.index.year == year) & (df.index.month == month)]
        if sub.empty:
            return None
        return sub.iloc[0]
    except Exception:
        return None

def _get_month_last_trade(df: pd.DataFrame, year: int, month: int):
    try:
        sub = df[(df.index.year == year) & (df.index.month == month)]
        if sub.empty:
            return None
        return sub.iloc[-1]
    except Exception:
        return None

def seasonal_backtest(ticker: str, buy_month: int, sell_month: int, years: int = 10) -> Dict:
    """
    Simulate seasonal trades for `ticker`.
    Buy at the first trading day 'Open' of buy_month and sell at the last trading day 'Close' of sell_month.
    If sell_month <= buy_month, sell occurs in next calendar year.
    Returns dict with per-year trade results, summary metrics, and equity curve.
    """
    tk = yf.Ticker(ticker)
    period = f"{years+1}y"  # +1 to allow last-year sells when needed
    hist = tk.history(period=period, interval="1d", actions=False).sort_index()
    if hist is None or hist.empty:
        return {"error": "No historical data"}

    hist = hist[['Open','High','Low','Close','Volume']].copy()
    hist.index = pd.to_datetime(hist.index)

    # determine available years (based on monthly resample)
    years_available = sorted(list(set(hist.index.year)))
    # We'll consider trades for years where both buy and sell dates are present
    trades = []
    # We will simulate for each year in years_available except possibly last if sell is next year and data missing
    for y in years_available:
        buy_year = y
        sell_year = y if sell_month > buy_month else (y + 1)
        # find buy trade
        buy_row = _get_month_first_trade(hist, buy_year, buy_month)
        sell_row = _get_month_last_trade(hist, sell_year, sell_month)
        if buy_row is None or sell_row is None:
            # skip incomplete trade
            continue
        buy_price = float(buy_row['Open'])  # buy at open
        sell_price = float(sell_row['Close'])  # sell at close
        ret = (sell_price - buy_price) / buy_price
        trades.append({
            'buy_date': buy_row.name,
            'sell_date': sell_row.name,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'return': ret,
            'buy_year': buy_year,
            'sell_year': sell_year
        })

    if not trades:
        return {"error": "No complete trades found for specified months and history length."}

    # compute metrics
    df_tr = pd.DataFrame(trades)
    n = len(df_tr)
    wins = (df_tr['return'] > 0).sum()
    win_rate = wins / n
    avg_ret = df_tr['return'].mean()
    median_ret = df_tr['return'].median()
    std_ret = df_tr['return'].std()
    # total return across trades (compounded)
    cum_return = np.prod(1 + df_tr['return'].values) - 1
    # duration in years as (last sell_date - first buy_date).days / 365
    duration_days = (df_tr['sell_date'].max() - df_tr['buy_date'].min()).days
    duration_years = max(duration_days / 365.25, 1/365.25)
    CAGR = (1 + cum_return) ** (1 / duration_years) - 1
    # equity curve: cumulative product of (1+ret)
    cumulative_vals = (1 + df_tr['return']).cumprod()
    # max drawdown on cumulative series
    peaks = cumulative_vals.cummax()
    drawdowns = (cumulative_vals - peaks) / peaks
    max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0

    # Prepare time-series equity curve with dates = sell_date and value normalized to 1
    equity = pd.Series(data=cumulative_vals.values, index=df_tr['sell_date']).sort_index()

    summary = {
        'ticker': ticker,
        'buy_month': buy_month,
        'sell_month': sell_month,
        'n_trades': n,
        'wins': int(wins),
        'win_rate': float(win_rate),
        'avg_return': float(avg_ret),
        'median_return': float(median_ret),
        'std_return': float(std_ret) if not np.isnan(std_ret) else 0.0,
        'total_compounded_return': float(cum_return),
        'CAGR': float(CAGR),
        'max_drawdown': float(max_drawdown),
        'trade_table': df_tr,
        'equity_curve': equity
    }
    return summary

def portfolio_backtest_from_alloc(allocs: List[Dict], buy_month:int, sell_month:int, years:int=10) -> Dict:
    """
    Simple portfolio-level backtest using allocation 'allocs' produced by allocation.allocate_budget.
    allocs: list of dicts each with keys: ticker, qty, current_price, invested
    This function:
      - For each ticker, runs seasonal_backtest(ticker, buy_month, sell_month)
      - Converts per-trade returns into monetary returns using invested proportion of initial budget
      - Builds portfolio equity curve by summing allocations' equity curves (aligned by sell_date)
    Returns:
      - portfolio equity Series (indexed by sell dates)
      - summary metrics (CAGR, total return, max drawdown)
      - per-ticker backtests for inspection
    Note: This is a simple yearly aggregation, assumes re-investment each year into same allocation.
    """
    # Determine budget from allocs
    budget = sum([float(a.get('invested',0) or 0) for a in allocs])
    per_ticker_results = {}
    # Gather per-ticker series in money terms (starting value = invested amount)
    money_series_list = []
    for a in allocs:
        t = a['ticker']
        invested = float(a.get('invested',0) or 0)
        if invested <= 0:
            continue
        bt = seasonal_backtest(t, buy_month, sell_month, years=years)
        if 'error' in bt:
            continue
        eq = bt['equity_curve']
        # scale equity (currently normalized to starting 1) to invested amount
        scaled = eq * invested
        per_ticker_results[t] = {'summary': bt, 'scaled_equity': scaled}
        money_series_list.append(scaled)

    if not money_series_list:
        return {'error': 'No valid per-ticker backtests (no price data or insufficient history).'}

    # Align all series by union of indices and sum
    all_idx = sorted(set().union(*[s.index for s in money_series_list]))
    df_port = pd.DataFrame(index=all_idx)
    for t, res in per_ticker_results.items():
        s = res['scaled_equity']
        df_port[t] = s.reindex(all_idx).ffill().fillna(method='ffill').fillna(0)

    # portfolio equity = sum across tickers
    df_port['portfolio_value'] = df_port.sum(axis=1)
    # normalize to 1 at first date (if budget >0)
    if budget <= 0:
        return {'error': 'Budget is zero in allocation.'}
    # compute returns series (monetary), but for metrics compute relative
    portfolio_series = df_port['portfolio_value']
    # compute CAGR based on first and last index dates
    start_val = float(portfolio_series.iloc[0])
    end_val = float(portfolio_series.iloc[-1])
    total_return = (end_val / start_val) - 1 if start_val>0 else 0.0
    duration_days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
    duration_years = max(duration_days / 365.25, 1/365.25)
    CAGR = (1 + total_return) ** (1 / duration_years) - 1 if total_return > -1 else -1.0
    # compute max drawdown
    cum = portfolio_series.cummax()
    drawdowns = (portfolio_series - cum) / cum
    max_dd = drawdowns.min()
    result = {
        'portfolio_series': portfolio_series,
        'total_return': float(total_return),
        'CAGR': float(CAGR),
        'max_drawdown': float(max_dd),
        'per_ticker': per_ticker_results
    }
    return result
