# allocation.py
# Budget -> allocation engine
# Input: candidates (list of dicts returned by pipeline.analyze_universe_with_fundamentals)
# Output: allocation dict with qty per ticker, invested amounts, leftover cash.

import pandas as pd
import numpy as np

# default caps
MAX_POS_PERCENT = 0.10   # max 10% per stock
MAX_SECTOR_PERCENT = 0.25  # max 25% per sector

def allocate_budget(candidates, budget, max_pos_percent=MAX_POS_PERCENT, max_sector_percent=MAX_SECTOR_PERCENT):
    """
    candidates: list of dicts, each must include:
      - 'ticker'
      - 'score'
      - 'fundamentals' -> may include 'sector' (string)
      - 'current_price' -> float (we'll try to add it before calling)
    budget: float (INR)
    Returns:
      {
        'allocs': [ {ticker, sector, score, current_price, alloc_amt, qty, invested} ],
        'leftover': float,
        'total_invested': float
      }
    """
    # sanity
    if not candidates or budget <= 0:
        return {'allocs': [], 'leftover': budget, 'total_invested': 0.0}

    # build DF from candidates
    rows = []
    for c in candidates:
        # expect current_price present; if not, skip that candidate
        cp = c.get('current_price')
        if cp is None or cp <= 0:
            continue
        sector = c.get('fundamentals', {}).get('sector') or 'Unknown'
        rows.append({
            'ticker': c['ticker'],
            'score': float(c.get('score', 0)),
            'sector': sector,
            'current_price': float(cp)
        })
    if not rows:
        return {'allocs': [], 'leftover': budget, 'total_invested': 0.0}

    df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    # initial raw weight proportional to score
    df['raw_w'] = df['score'] / (df['score'].sum() + 1e-9)
    df['alloc_amt'] = df['raw_w'] * budget
    # enforce per-stock cap
    df['max_alloc_amt'] = budget * max_pos_percent
    df['alloc_amt'] = df[['alloc_amt', 'max_alloc_amt']].min(axis=1)
    # enforce sector caps iteratively
    # if any sector exceeds cap, scale down proportional within that sector
    sector_totals = df.groupby('sector')['alloc_amt'].sum().to_dict()
    for sector, tot in sector_totals.items():
        cap = budget * max_sector_percent
        if tot > cap and tot > 0:
            scale = cap / tot
            df.loc[df['sector'] == sector, 'alloc_amt'] = df.loc[df['sector'] == sector, 'alloc_amt'] * scale

    # convert to integer quantities
    df['qty'] = (df['alloc_amt'] / df['current_price']).apply(np.floor).astype(int)
    df['invested'] = df['qty'] * df['current_price']
    total_invested = df['invested'].sum()
    leftover = float(budget - total_invested)
    allocs = df[['ticker','sector','score','current_price','alloc_amt','qty','invested']].to_dict('records')
    return {'allocs': allocs, 'leftover': leftover, 'total_invested': float(total_invested)}
