# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yfinance>=0.2.40",
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.8.0",
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import argparse

# Evitiamo di scrivere su portfolio.json
os.environ["YF_CONSENT"] = "1"

# Importa i moduli da aegis_momentum
sys.path.append(os.path.dirname(__file__))
from aegis_momentum import (
    UNIVERSE, MarketDataProvider, MomentumScorer, PortfolioAllocator, RiskManager, CASH_PROXY, INITIAL_CAPITAL
)

def download_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Scaricamento dati storici (dal 2013 in poi)...")
    tickers = list(UNIVERSE.keys())
    # Scarica dal 2013-01-01 per garantire dati per il primo ribilanciamento 12 mesi dopo
    data = yf.download(tickers, start="2013-01-01", auto_adjust=True, progress=False, threads=False)
    
    if data.empty:
        raise RuntimeError("Nessun dato ricevuto da Yahoo Finance.")
        
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
        
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    returns = prices.pct_change().dropna()
    
    # Forward fill per gestire giorni festivi asincroni
    prices = prices.ffill()
    
    print(f"Dati caricati. {len(prices)} giorni da {prices.index[0].date()} a {prices.index[-1].date()}")
    return prices, returns

def run_backtest(years: int, prices: pd.DataFrame, returns: pd.DataFrame):
    print(f"\n{'='*50}\nEsecuzione Backtest: {years} Anni\n{'='*50}")
    
    end_date = prices.index[-1]
    start_date = end_date - pd.DateOffset(years=years)
    
    # Troviamo l'indice più vicino alla data di inizio
    # Deve esserci almeno 1 anno di dati prima della data di inizio (per il lookback)
    start_idx = prices.index.searchsorted(start_date)
    if start_idx < 252:
        start_idx = 252  # Almeno 1 anno di dati
        print(f"ATTENZIONE: Dati storici insufficienti per {years} anni. Partenza dal {prices.index[start_idx].date()}")
    else:
        print(f"Inizio simulazione dal: {prices.index[start_idx].date()}")
        
    # Stato del portafoglio
    portfolio = {
        "cash": INITIAL_CAPITAL,
        "holdings": [],
        "peak_value": INITIAL_CAPITAL
    }
    
    value_history = []
    spy_history = []
    
    spy_start_price = prices['SPY'].iloc[start_idx]
    
    # Loop settimanale (ogni 5 giorni lavorativi)
    for i in range(start_idx, len(prices), 5):
        current_date = prices.index[i]
        
        # Taglia i dati fino ad oggi
        prices_slice = prices.iloc[:i+1]
        returns_slice = returns.iloc[:i+1]
        current_prices = prices_slice.iloc[-1]
        
        # Aggiorna valore
        current_value = portfolio["cash"] + sum(
            h["quantity"] * current_prices.get(h["ticker"], 0)
            for h in portfolio["holdings"]
            if h["ticker"] in current_prices.index
        )
        
        # Risk Management (Circuit Breaker)
        risk_mgr = RiskManager(portfolio)
        
        # Silenzia temporaneamente l'output per evitare spam nel backtest
        import builtins
        original_print = builtins.print
        builtins.print = lambda *a, **k: None
        
        cb_level = risk_mgr.check_portfolio_drawdown(current_value)
        
        # Stop loss check
        stops = risk_mgr.check_position_stops(portfolio["holdings"], current_prices)
        for ticker in stops:
            holding = next((h for h in portfolio["holdings"] if h["ticker"] == ticker), None)
            if holding and ticker in current_prices.index:
                # Esegui vendita (simulata)
                proceeds = holding["quantity"] * current_prices[ticker] * 0.999 # 0.1% slippage/fee
                portfolio["cash"] += proceeds
                portfolio["holdings"] = [h for h in portfolio["holdings"] if h["ticker"] != ticker]
                
        # Ricalcola valore dopo stop loss
        current_value = portfolio["cash"] + sum(
            h["quantity"] * current_prices.get(h["ticker"], 0)
            for h in portfolio["holdings"]
            if h["ticker"] in current_prices.index
        )
        
        # Scoring & Allocazione
        data_provider = MarketDataProvider(list(UNIVERSE.keys()))
        data_provider.set_data_slice(prices_slice, returns_slice)
        
        # Scoring & Allocazione
        scorer = MomentumScorer(data_provider)
        scores = scorer.compute_scores()
        
        allocator = PortfolioAllocator(data_provider, scores)
        target_weights = allocator.allocate()
        
        target_weights = risk_mgr.apply_circuit_breaker(target_weights, cb_level)
        
        # Ribilanciamento (con slippage 0.1%)
        current_tickers = {h["ticker"] for h in portfolio["holdings"]}
        target_tickers = {t for t, w in target_weights.items() if w > 0.01 and t != CASH_PROXY}

        # Vendite
        for holding in list(portfolio["holdings"]):
            ticker = holding["ticker"]
            if ticker not in target_tickers and ticker in current_prices.index:
                proceeds = holding["quantity"] * current_prices[ticker] * 0.999
                portfolio["cash"] += proceeds
                portfolio["holdings"] = [h for h in portfolio["holdings"] if h["ticker"] != ticker]

        # Acquisti
        target_values = {
            ticker: current_value * weight
            for ticker, weight in target_weights.items()
            if ticker != CASH_PROXY and weight > 0.01
        }
        
        for holding in list(portfolio["holdings"]):
            ticker = holding["ticker"]
            if ticker in target_values and ticker in current_prices.index:
                current_val = holding["quantity"] * current_prices[ticker]
                target_val = target_values[ticker]
                diff = target_val - current_val
                if diff < -1.0:
                    qty_to_sell = abs(diff) / current_prices[ticker]
                    proceeds = qty_to_sell * current_prices[ticker] * 0.999
                    portfolio["cash"] += proceeds
                    holding["quantity"] -= qty_to_sell
                    target_values[ticker] = 0
                elif diff > 1.0:
                    target_values[ticker] = diff
                else:
                    target_values[ticker] = 0

        for ticker, amount in target_values.items():
            if amount > 1.0 and ticker in current_prices.index:
                buy_amount = min(amount, portfolio["cash"])
                if buy_amount > 0.50:
                    qty = (buy_amount * 0.999) / current_prices[ticker] # 0.1% slippage
                    portfolio["cash"] -= buy_amount
                    # Add holding
                    existing = [h for h in portfolio["holdings"] if h["ticker"] == ticker]
                    if existing:
                        existing[0]["quantity"] += qty
                    else:
                        portfolio["holdings"].append({"ticker": ticker, "quantity": qty, "buy_price": current_prices[ticker]})

        # Ripristina print
        builtins.print = original_print
        
        # Valore finale step
        final_value = portfolio["cash"] + sum(
            h["quantity"] * current_prices.get(h["ticker"], 0)
            for h in portfolio["holdings"]
            if h["ticker"] in current_prices.index
        )
        if final_value > portfolio["peak_value"]:
            portfolio["peak_value"] = final_value
            
        value_history.append((current_date, final_value))
        
        # Benchmark SPY
        spy_val = INITIAL_CAPITAL * (current_prices['SPY'] / spy_start_price)
        spy_history.append((current_date, spy_val))
        
        sys.stdout.write(f"\rProgresso: {current_date.date()} | Valore: ${final_value:.2f}")
        sys.stdout.flush()
        
    print("\nCompletato.")
    
    # Statistiche
    final_val = value_history[-1][1]
    tot_ret = (final_val / INITIAL_CAPITAL) - 1
    cagr = (final_val / INITIAL_CAPITAL) ** (1/years) - 1
    
    df_val = pd.DataFrame(value_history, columns=['date', 'value']).set_index('date')
    roll_max = df_val['value'].cummax()
    drawdowns = (df_val['value'] - roll_max) / roll_max
    max_dd = drawdowns.min()
    
    print(f"\n📊 RISULTATI ({years} ANNI):")
    print(f"Rendimento Totale: {tot_ret:+.2%}")
    print(f"CAGR (Annuale):    {cagr:+.2%}")
    print(f"Max Drawdown:      {max_dd:+.2%}")
    
    return df_val, pd.DataFrame(spy_history, columns=['date', 'value']).set_index('date'), cagr, max_dd

if __name__ == "__main__":
    prices, returns = download_all_data()
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    for i, years in enumerate([1, 5, 10]):
        df_aegis, df_spy, cagr, max_dd = run_backtest(years, prices, returns)
        
        ax = axes[i]
        ax.plot(df_aegis.index, df_aegis['value'], label=f'Aegis Momentum (CAGR: {cagr:.1%}, MDD: {max_dd:.1%})', color='#818cf8', linewidth=2)
        ax.plot(df_spy.index, df_spy['value'], label='S&P 500 Buy & Hold', color='#64748b', linewidth=1.5, alpha=0.7)
        ax.set_title(f'Backtest: Ultimi {years} Anni')
        ax.legend()
        ax.grid(color='#334155', alpha=0.3)
        ax.set_ylabel('Valore Portafoglio ($)')
        
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Grafico salvato come 'backtest_results.png'")
