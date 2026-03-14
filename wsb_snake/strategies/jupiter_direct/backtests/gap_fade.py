#!/usr/bin/env python3
"""
GAP FADE MODE - FADE THE GAP, DON'T CHASE IT
=============================================
When a stock gaps 5%+, it often FADES (fills the gap).
Smart money sells into strength, buys into weakness.

Gap UP = BUY PUTS (fade)
Gap DOWN = BUY CALLS (fade)
"""

import os, json, time, logging, requests
from datetime import datetime, timedelta

STARTING_CAPITAL = 5000
LEVERAGE = 12  # Slightly less aggressive
MIN_GAP = 5.0
STOP = -0.40  # Tighter stop
TARGET = 1.50  # +150%

ALP_KEY = os.environ.get('ALPACA_API_KEY')
ALP_SEC = os.environ.get('ALPACA_SECRET_KEY')
ALP_H = {'APCA-API-KEY-ID': ALP_KEY, 'APCA-API-SECRET-KEY': ALP_SEC}

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def get_gaps(date):
    tickers = ['TSLA','NVDA','AMD','COIN','MARA','RIOT','PLTR','SOFI','GME','AMC',
               'HOOD','SMCI','ARM','SNAP','SQ','SHOP','RBLX','DKNG','META','GOOGL']
    gaps = []
    for t in tickers:
        try:
            r = requests.get(f'https://data.alpaca.markets/v2/stocks/{t}/bars',
                headers=ALP_H, params={'start':f'{date}T00:00:00-04:00',
                'end':f'{date}T09:30:00-04:00','timeframe':'1Day','limit':2}, timeout=3)
            prev = r.json().get('bars',[])
            if not prev: continue
            pc = prev[-1]['c']
            
            r2 = requests.get(f'https://data.alpaca.markets/v2/stocks/{t}/bars',
                headers=ALP_H, params={'start':f'{date}T09:30:00-04:00',
                'end':f'{date}T09:35:00-04:00','timeframe':'1Min','limit':3}, timeout=3)
            op = r2.json().get('bars',[])
            if not op: continue
            
            gap = ((op[0]['o'] - pc) / pc) * 100
            if abs(gap) >= MIN_GAP:
                # FADE THE GAP - opposite direction!
                gaps.append({'t':t,'gap':gap,'dir':'PUTS' if gap>0 else 'CALLS','prev_close':pc})
        except: continue
        time.sleep(0.02)
    gaps.sort(key=lambda x: abs(x['gap']), reverse=True)
    return gaps

def get_bars(t, date, s, e):
    try:
        r = requests.get(f'https://data.alpaca.markets/v2/stocks/{t}/bars',
            headers=ALP_H, params={'start':f'{date}T{s}:00-04:00',
            'end':f'{date}T{e}:00-04:00','timeframe':'1Min','limit':400}, timeout=8)
        return r.json().get('bars',[])
    except: return []

def run():
    log.info("="*60)
    log.info("GAP FADE MODE - FADE THE OVEREXTENSION")
    log.info("="*60)
    log.info(f"Gap UP > {MIN_GAP}% = BUY PUTS (fade)")
    log.info(f"Gap DOWN > {MIN_GAP}% = BUY CALLS (fade)")
    log.info(f"${STARTING_CAPITAL:,} | {LEVERAGE}x | Stop {STOP*100:.0f}% | +{TARGET*100:.0f}%")
    log.info("="*60)
    
    start = datetime(2026, 2, 24)
    end = datetime(2026, 3, 12)
    capital = STARTING_CAPITAL
    trades = []
    
    cur = start
    while cur <= end:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue
        
        ds = cur.strftime('%Y-%m-%d')
        log.info(f"\n{ds} | ${capital:,.0f}")
        
        gaps = get_gaps(ds)
        if not gaps:
            log.info("  No 5%+ gaps")
            cur += timedelta(days=1)
            continue
        
        g = gaps[0]
        ticker, gap, direction, prev_close = g['t'], g['gap'], g['dir'], g['prev_close']
        log.info(f"  {ticker} GAP {gap:+.1f}% -> FADE with {direction}")
        
        # Entry at 10:00 AM after initial move
        bars = get_bars(ticker, ds, '09:30', '10:00')
        if len(bars) < 5:
            cur += timedelta(days=1)
            continue
        
        entry = bars[-1]['c']
        opt_cost = entry * 0.002
        contracts = max(1, int(capital / (opt_cost * 100)))
        position = contracts * opt_cost * 100
        
        log.info(f"  FADE ENTRY @ ${entry:.2f} | {contracts} contracts = ${position:,.0f}")
        log.info(f"  Target: Gap fill to ${prev_close:.2f}")
        
        # Simulate rest of day
        rest = get_bars(ticker, ds, '10:00', '16:00')
        exit_p, reason = entry, 'EOD'
        
        for b in rest:
            # For PUTS (fading gap up), we profit when price DROPS
            # For CALLS (fading gap down), we profit when price RISES
            if direction == 'PUTS':
                # Profiting when price drops from entry
                best_mv = (entry - b['l']) / entry
                cur_mv = (entry - b['c']) / entry
            else:
                # Profiting when price rises from entry
                best_mv = (b['h'] - entry) / entry
                cur_mv = (b['c'] - entry) / entry
            
            opt_best = best_mv * LEVERAGE
            opt_cur = cur_mv * LEVERAGE
            
            if opt_best >= TARGET:
                if direction == 'PUTS':
                    exit_p = entry * (1 - TARGET/LEVERAGE)
                else:
                    exit_p = entry * (1 + TARGET/LEVERAGE)
                reason = f'+{int(TARGET*100)}%!'
                log.info(f"  >>> GAP FADE SUCCESS! +{int(TARGET*100)}% <<<")
                break
            
            if opt_cur <= STOP:
                exit_p = b['c']
                reason = 'STOP'
                break
            
            exit_p = b['c']
        
        # Calculate P&L based on direction
        if direction == 'PUTS':
            stk = (entry - exit_p) / entry
        else:
            stk = (exit_p - entry) / entry
        
        opt_pnl = stk * LEVERAGE
        pnl = position * opt_pnl
        capital += pnl
        
        log.info(f"  EXIT ({reason}) ${exit_p:.2f} | Stock {stk*100:+.1f}% -> Opt {opt_pnl*100:+.0f}%")
        log.info(f"  P&L: ${pnl:+,.0f} | CAPITAL: ${capital:,.0f}")
        
        trades.append({'date':ds,'t':ticker,'dir':direction,'gap':gap,
            'entry':entry,'exit':exit_p,'reason':reason,'opt':opt_pnl*100,'pnl':pnl})
        
        time.sleep(0.2)
        cur += timedelta(days=1)
    
    ret = ((capital - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
    wins = len([x for x in trades if x['pnl'] > 0])
    
    log.info("\n" + "="*60)
    log.info("GAP FADE RESULTS")
    log.info("="*60)
    log.info(f"${STARTING_CAPITAL:,} -> ${capital:,.0f}")
    log.info(f"RETURN: {ret:+.1f}%")
    if trades:
        log.info(f"Trades: {len(trades)} | Wins: {wins} | Rate: {wins/len(trades)*100:.0f}%")
    log.info("="*60)
    
    log.info("\nTRADE LOG:")
    for t in trades:
        w = "W" if t['pnl'] > 0 else "L"
        log.info(f"  {t['date']} {t['t']:5} {t['dir']:5} gap:{t['gap']:+.1f}% opt:{t['opt']:+.0f}% ${t['pnl']:+,.0f} [{w}]")
    
    log.info("\n" + "="*60)
    log.info("VS REDDIT:")
    log.info("u/imsuffi: +733%")
    log.info("u/mneymaker: +301%")
    log.info(f"GAP FADE: {ret:+.1f}%")
    
    if ret >= 200: log.info("\n>>> REDDIT CRUSHED <<<")
    elif ret >= 100: log.info("\n>>> COMPETITIVE <<<")
    elif ret >= 0: log.info("\n>>> PROFITABLE <<<")
    else: log.info("\n>>> RETHINK STRATEGY <<<")
    log.info("="*60)
    
    os.makedirs('/home/ubuntu/wsb-snake/backtest/gap_fade', exist_ok=True)
    with open('/home/ubuntu/wsb-snake/backtest/gap_fade/results.json', 'w') as f:
        json.dump({'return':ret,'capital':capital,'trades':trades}, f, indent=2, default=str)

if __name__ == '__main__':
    run()
