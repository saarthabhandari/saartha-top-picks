# app.py — Saartha's top picks (final polished)
# Overwrite your existing app.py with this file.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from calendar import month_name
from typing import List, Dict
import io, json, time

# local modules (must exist in same folder)
import pipeline
import allocation
import backtest
import sentiment_news

# -------- Page config & styles --------
st.set_page_config(page_title="Saartha's top picks", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
      --bg: #f6f9fc;
      --card: #ffffff;
      --muted: #6b7280;
      --accent: #0ea5a2;
      --accent-2: #2563eb;
      --good: #16a34a;
      --bad: #dc2626;
    }
    .stApp { background-color: var(--bg); }
    .header-row { display:flex; align-items:center; justify-content:space-between; }
    .title { font-size:28px; font-weight:700; margin:0; }
    .subtitle { color: var(--muted); margin-top:3px; margin-bottom:8px; }
    .card { background: var(--card); padding: 14px; border-radius:10px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
    .metric { font-size:20px; font-weight:700; }
    .small-muted { color: var(--muted); font-size:13px; }
    .btn { background: linear-gradient(90deg,var(--accent-2),var(--accent)); color: white; border: none; padding: 8px 12px; border-radius:8px; }
    </style>
    """, unsafe_allow_html=True
)

# -------- Header --------
col1, col2 = st.columns([4,1])
with col1:
    st.markdown('<div class="title">Saartha\'s top picks</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Seasonality + Fundamentals + Optional Technical Filters</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f"**{datetime.now().strftime('%Y-%m-%d')}**")

# -------- Sidebar Controls --------
with st.sidebar:
    st.header("Controls")
    budget = st.number_input("Budget (₹)", min_value=1000, value=10000, step=500, format="%d")
    mode = st.selectbox("Mode", ["Balanced", "Conservative", "Aggressive"], index=0)
    years = st.selectbox("History (years)", [5,7,10], index=2)
    min_score = st.slider("Min seasonality score", 0, 100, 65)
    sel_month = st.selectbox("Buy month", options=list(range(1,13)), format_func=lambda x: month_name[x], index=11)
    st.markdown("---")
    st.write("Technical filters (optional)")
    use_ind = st.checkbox("Enable technical filters", value=False)
    rsi_threshold = st.slider("RSI ≤", 10, 60, 40) if use_ind else 40
    price_above_sma50 = st.checkbox("Price > 50-day SMA", value=False) if use_ind else False
    price_above_sma200 = st.checkbox("Price > 200-day SMA", value=False) if use_ind else False
    st.markdown("---")
    run_btn = st.button("Run")
    if st.button("Clear cache & refresh"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()
    st.markdown("<div class='small-muted'>Ensure universe.csv exists (column: ticker, Yahoo format e.g. RELIANCE.NS)</div>", unsafe_allow_html=True)

# -------- Mode thresholds --------
if mode == "Conservative":
    thresholds = {
        'marketcap_min': 50000 * 1e7, 'min_avg_volume': 100_000, 'max_debt_to_equity': 1.0,
        'score_high': 85, 'pctpos_high': 85, 'years_high': 9, 'p_high': 0.02,
        'score_med': 70, 'pctpos_med': 75, 'years_med': 7, 'p_med': 0.05
    }
elif mode == "Aggressive":
    thresholds = {
        'marketcap_min': 5000 * 1e7, 'min_avg_volume': 20_000, 'max_debt_to_equity': 3.0,
        'score_high': 75, 'pctpos_high': 75, 'years_high': 6, 'p_high': 0.05,
        'score_med': 60, 'pctpos_med': 65, 'years_med': 5, 'p_med': 0.1
    }
else:
    thresholds = {
        'marketcap_min': 20000 * 1e7, 'min_avg_volume': 50_000, 'max_debt_to_equity': 2.0,
        'score_high': 80, 'pctpos_high': 80, 'years_high': 8, 'p_high': 0.05,
        'score_med': 65, 'pctpos_med': 70, 'years_med': 6, 'p_med': 0.1
    }

# -------- Utilities --------
@st.cache_data(ttl=300)
def load_universe(path="universe.csv") -> List[str]:
    try:
        df = pd.read_csv(path)
        tickers = [t.strip() for t in df['ticker'].tolist() if str(t).strip()]
        return tickers
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_latest_prices(tickers: List[str]) -> Dict[str, float]:
    out = {t: None for t in tickers}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="7d", interval="1d")
            if hist is not None and not hist.empty:
                if 'Close' in hist.columns and not hist['Close'].dropna().empty:
                    out[t] = float(hist['Close'].dropna().iloc[-1])
                elif 'Adj Close' in hist.columns and not hist['Adj Close'].dropna().empty:
                    out[t] = float(hist['Adj Close'].dropna().iloc[-1])
        except Exception:
            out[t] = None
    return out

def months_between(start:int, end:int):
    seq = [start]
    m = start
    while m != end:
        m = (m % 12) + 1
        seq.append(m)
    return seq

def compute_trade_targets(candidate: dict, current_price: float, buy_month_override: int = None, sell_month_override: int = None):
    out = {'buy_month':None,'sell_month':None,'hold_desc':None,'expected_return_pct':None,'target_sell_price':None,'suggested_buy_limit':None}
    if not candidate or current_price is None:
        return out
    ms = candidate.get('monthly_stats')
    if ms is None or ms.empty:
        return out
    buy_month = buy_month_override if buy_month_override else (candidate.get('buy_months') or [None])[0]
    sell_month = sell_month_override if sell_month_override else (candidate.get('peak_months') or [None])[0]
    if buy_month is None or sell_month is None:
        return out
    buy_month = int(buy_month); sell_month = int(sell_month)
    seq = months_between(buy_month, sell_month)
    out['buy_month'] = buy_month; out['sell_month'] = sell_month
    out['hold_desc'] = f"{month_name[buy_month]} → {month_name[sell_month]} ({len(seq)} month(s))"
    cum = 1.0
    for m in seq:
        row = ms[ms['month']==m]
        r = float(row['avg_ret'].iloc[0]) if (not row.empty and not pd.isna(row['avg_ret'].iloc[0])) else 0.0
        cum *= (1 + r)
    expected_return = cum - 1.0
    out['expected_return_pct'] = round(expected_return*100,2)
    out['target_sell_price'] = round(current_price * (1 + expected_return), 2)
    buy_row = ms[ms['month']==buy_month]
    std_buy = float(buy_row['std'].iloc[0]) if (not buy_row.empty and not pd.isna(buy_row['std'].iloc[0])) else 0.02
    buffer = min(0.06, max(0.01, std_buy))
    out['suggested_buy_limit'] = round(current_price * (1 - buffer), 2)
    return out

@st.cache_data(ttl=300)
def compute_indicators_cached(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d")
        if hist is None or hist.empty:
            return {'rsi14':None,'sma50':None,'sma200':None,'last_close':None}
        close = hist['Close'].dropna().astype(float)
        last_close = float(close.iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        delta = close.diff().dropna()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi14 = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
        return {'rsi14': rsi14, 'sma50': sma50, 'sma200': sma200, 'last_close': last_close}
    except Exception:
        return {'rsi14':None,'sma50':None,'sma200':None,'last_close':None}

@st.cache_data(ttl=1800)
def compute_candidates(tickers: List[str], years_val:int, thr_score:int, thresholds_dict:dict):
    return pipeline.analyze_universe_with_fundamentals(tickers, years=years_val, thresholds=thresholds_dict, min_score=thr_score)

# -------- Main flow --------
tickers = load_universe()
if not tickers:
    st.warning("universe.csv not found or empty. Run generate_universe.py or add a file named universe.csv (column 'ticker').")
    st.stop()

if run_btn:
    with st.spinner("Finding candidates..."):
        candidates = compute_candidates(tickers, years, min_score, thresholds)
else:
    candidates = compute_candidates(tickers, years, min_score, thresholds)

# header metrics
r1, r2, r3, r4 = st.columns([1,1,1,2])
r1.metric("Universe", len(tickers))
r2.metric("Candidates", len(candidates))
r3.metric("Budget (₹)", int(budget))
r4.markdown(f"**Mode:** {mode} • **History:** {years} yrs • **Min score:** {min_score}")

if not candidates:
    st.info("No candidates found. Try Aggressive mode or lower min score.")
    st.stop()

# fetch latest prices
candidate_tickers = [c['ticker'] for c in candidates]
with st.spinner("Fetching latest prices..."):
    price_map = get_latest_prices(candidate_tickers)

# attach price and compute display rows
for c in candidates:
    c['current_price'] = price_map.get(c['ticker'])

display_rows = []
indicator_cache = {}

for c in candidates:
    if sel_month not in (c.get('buy_months') or []):
        continue
    price = c.get('current_price')
    if price is None:
        continue
    targets = compute_trade_targets(c, price)
    # indicators
    if use_ind:
        ind = compute_indicators_cached(c['ticker'])
        indicator_cache[c['ticker']] = ind
        rsi_ok = (ind['rsi14'] is not None and ind['rsi14'] <= rsi_threshold)
        sma50_ok = True if not price_above_sma50 else (ind['sma50'] is not None and ind['last_close'] is not None and ind['last_close'] > ind['sma50'])
        sma200_ok = True if not price_above_sma200 else (ind['sma200'] is not None and ind['last_close'] is not None and ind['last_close'] > ind['sma200'])
        if not (rsi_ok and sma50_ok and sma200_ok):
            continue
    row = {
        'Ticker': c['ticker'],
        'Score': round(c['score'],2),
        'Reliability': c.get('reliability_label','-'),
        'Price': price,
        'Buy limit': targets.get('suggested_buy_limit'),
        'Target sell': targets.get('target_sell_price'),
        'Expected %': targets.get('expected_return_pct'),
        'Hold': targets.get('hold_desc'),
        'Buy months': ", ".join(map(str,c.get('buy_months') or [])),
        'Sell months': ", ".join(map(str,c.get('peak_months') or []))
    }
    if use_ind:
        ind = indicator_cache.get(c['ticker'])
        row.update({
            'RSI14': round(ind['rsi14'],2) if ind and ind['rsi14'] is not None else None,
            'SMA50': round(ind['sma50'],2) if ind and ind['sma50'] is not None else None,
            'SMA200': round(ind['sma200'],2) if ind and ind['sma200'] is not None else None
        })
    display_rows.append(row)

df_display = pd.DataFrame(display_rows)
st.markdown("### Ideas")
if df_display.empty:
    st.info(f"No ideas for {month_name[sel_month]} with current filters.")
else:
    base_cols = ['Ticker','Score','Reliability','Price','Buy limit','Target sell','Expected %','Hold']
    if use_ind:
        base_cols += ['RSI14','SMA50','SMA200']
    st.dataframe(df_display[base_cols].style.format({"Price":"{:.2f}","Buy limit":"{:.2f}","Target sell":"{:.2f}","Expected %":"{:.2f}"}), height=420)
    csv_buf = io.StringIO(); df_display.to_csv(csv_buf, index=False)
    st.download_button("Download ideas CSV", csv_buf.getvalue().encode('utf-8'), file_name=f"{month_name[sel_month]}_ideas.csv")

# -------- Allocation (session_state) --------
st.markdown("---")
st.markdown("### Allocate")

if "alloc_res" not in st.session_state:
    st.session_state.alloc_res = None

if st.button("Allocate budget"):
    with st.spinner("Allocating..."):
        alloc_res = allocation.allocate_budget(candidates, budget)
        # enrich allocs
        for a in alloc_res['allocs']:
            cand = next((x for x in candidates if x['ticker']==a['ticker']), None)
            tgt = compute_trade_targets(cand, a.get('current_price')) if cand else {}
            a['suggested_buy_limit'] = tgt.get('suggested_buy_limit')
            a['target_sell_price'] = tgt.get('target_sell_price')
            a['expected_return_pct'] = tgt.get('expected_return_pct')
        st.session_state.alloc_res = alloc_res
        df_alloc = pd.DataFrame(alloc_res['allocs'])
        st.dataframe(df_alloc.style.format({"current_price":"{:.2f}","alloc_amt":"{:.2f}","invested":"{:.2f}","suggested_buy_limit":"{:.2f}","target_sell_price":"{:.2f}"}), height=300)
        st.success(f"Invested ₹{alloc_res['total_invested']:.2f} • Leftover ₹{alloc_res['leftover']:.2f}")
        csv_buf = io.StringIO(); df_alloc.to_csv(csv_buf, index=False)
        st.download_button("Download allocation CSV", csv_buf.getvalue().encode('utf-8'), file_name="allocation.csv")

elif st.session_state.alloc_res:
    st.markdown("Allocation (saved)")
    df_alloc = pd.DataFrame(st.session_state.alloc_res['allocs'])
    st.dataframe(df_alloc.style.format({"current_price":"{:.2f}","alloc_amt":"{:.2f}","invested":"{:.2f}","suggested_buy_limit":"{:.2f}","target_sell_price":"{:.2f}"}), height=260)

else:
    st.info("Click 'Allocate budget' to convert budget into integer share quantities.")

# -------- Inspect & Backtest single candidate --------
st.markdown("---")
st.markdown("### Inspect & Backtest")
if df_display.size:
    sel = st.selectbox("Choose ticker to inspect", options=[r['Ticker'] for r in display_rows])
else:
    sel = None

if sel:
    chosen = next((c for c in candidates if c['ticker']==sel), None)
    price = chosen.get('current_price') if chosen else None
    st.markdown(f"**{sel}**  •  Price: ₹{price if price else '—'}  •  Score: {round(chosen['score'],2)}")
    ms = chosen.get('monthly_stats')
    if ms is not None and not ms.empty:
        ms_show = ms.copy(); ms_show['avg_ret_pct'] = ms_show['avg_ret'] * 100
        st.dataframe(ms_show[['month','avg_ret_pct','pct_positive','std','years_count']].rename(columns={'avg_ret_pct':'avg_ret (%)'}), height=220)
    hm = chosen.get('heatmap_matrix')
    if hm is not None and not hm.empty:
        fig, ax = plt.subplots(figsize=(10,3))
        sns.heatmap(hm*100, cmap="RdYlGn", center=0, ax=ax)
        ax.set_title("Monthly returns (%)")
        st.pyplot(fig)

    buy_default = int(chosen.get('buy_months')[0]) if chosen.get('buy_months') else sel_month
    sell_default = int(chosen.get('peak_months')[0]) if chosen.get('peak_months') else ((buy_default%12)+1)
    colA, colB = st.columns(2)
    buy_month = colA.selectbox("Buy month", options=list(range(1,13)), format_func=lambda x: month_name[x], index=buy_default-1)
    sell_month = colB.selectbox("Sell month", options=list(range(1,13)), format_func=lambda x: month_name[x], index=sell_default-1)

    if st.button("Run backtest for selected ticker"):
        with st.spinner("Backtesting..."):
            bt = backtest.seasonal_backtest(sel, buy_month, sell_month, years=years)
            if 'error' in bt:
                st.error(bt['error'])
            else:
                st.metric("Trades", bt['n_trades'])
                st.metric("Win rate", f"{bt['win_rate']*100:.1f}%")
                st.metric("CAGR", f"{bt['CAGR']*100:.2f}%")
                tr = bt['trade_table'].copy(); tr['return_pct'] = tr['return']*100
                st.dataframe(tr[['buy_date','sell_date','buy_price','sell_price','return_pct']].style.format({"buy_price":"{:.2f}","sell_price":"{:.2f}","return_pct":"{:.2f}"}), height=220)
                fig2, ax2 = plt.subplots(figsize=(8,3))
                eq = bt['equity_curve']; ax2.plot(eq.index, eq.values, marker='o'); ax2.set_title("Equity curve (normalized)")
                st.pyplot(fig2)
                csv_buf = io.StringIO(); tr.to_csv(csv_buf, index=False)
                st.download_button("Download trades CSV", csv_buf.getvalue().encode('utf-8'), file_name=f"{sel}_trades.csv")
                st.download_button("Download backtest JSON", json.dumps({
                    'summary': {k:v for k,v in bt.items() if k not in ['trade_table','equity_curve']},
                    'trades': json.loads(tr.to_json(orient='records', date_format='iso'))
                }, default=str, indent=2).encode('utf-8'), file_name=f"{sel}_backtest.json")

# -------- Portfolio backtest (uses session_state.alloc_res) --------
st.markdown("---")
st.markdown("### Portfolio backtest")
if st.session_state.get("alloc_res") and st.session_state.alloc_res.get('allocs'):
    if st.button("Run portfolio backtest"):
        with st.spinner("Running portfolio backtest..."):
            # use UI buy/sell selection if available, else default to sel_month & next
            bm = buy_month if 'buy_month' in locals() else sel_month
            sm = sell_month if 'sell_month' in locals() else ((bm%12)+1)
            alloc_res = st.session_state.alloc_res
            pb = backtest.portfolio_backtest_from_alloc(alloc_res['allocs'], bm, sm, years=years)
            if 'error' in pb:
                st.error(pb['error'])
            else:
                st.metric("Total return", f"{pb['total_return']*100:.2f}%")
                st.metric("CAGR", f"{pb['CAGR']*100:.2f}%")
                st.metric("Max drawdown", f"{pb['max_drawdown']*100:.2f}%")
                fig3, ax3 = plt.subplots(figsize=(8,3))
                series = pb['portfolio_series']; ax3.plot(series.index, series.values); ax3.set_title("Portfolio value")
                st.pyplot(fig3)
                csv_buf = io.StringIO(); series.reset_index().to_csv(csv_buf, index=False)
                st.download_button("Download portfolio CSV", csv_buf.getvalue().encode('utf-8'), file_name="portfolio_series.csv")
else:
    st.info("Allocate budget first to enable portfolio backtest (click 'Allocate budget' above).")

# -------- Footer --------
st.markdown("---")
st.caption("Historical-based suggestions only. Combine with fundamentals, news and your risk controls before trading.")

