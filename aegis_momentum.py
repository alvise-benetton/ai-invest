# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yfinance>=0.2.40",
#     "pandas>=2.0",
#     "numpy>=1.24",
# ]
# ///
"""
Aegis Momentum — Sistema di Gestione Patrimoniale Simulato
═══════════════════════════════════════════════════════════

Un motore di allocazione tattica multi-asset basato su Dual Momentum
potenziato, con protezione del capitale di livello istituzionale.

Strategia:
  1. Blended Momentum Score (3/6/12 mesi, skip-month, vol-adjusted)
  2. Selezione Top-3 + Filtro Assoluto (vs T-Bill)
  3. Inverse-Volatility Weighting
  4. Volatility Targeting (10% annualizzato)
  5. Circuit Breaker multi-livello

Esecuzione: uv run aegis_momentum.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

# Universo di investimento (16 ETF globali multi-asset)
UNIVERSE = {
    # Azionario
    "SPY": "S&P 500 (US Large Cap)",
    "QQQ": "Nasdaq 100 (US Growth)",
    "IWM": "Russell 2000 (US Small Cap)",
    "EFA": "MSCI EAFE (Developed ex-US)",
    "EEM": "MSCI EM (Emerging Markets)",
    "VNQ": "Vanguard Real Estate (REITs)",
    "VT":  "Vanguard Total World Stock",
    # Commodities
    "GLD": "SPDR Gold Shares",
    "DBC": "Invesco DB Commodity",
    # Obbligazionario
    "AGG": "iShares Core US Agg Bond",
    "TLT": "iShares 20+ Year Treasury",
    "IEF": "iShares 7-10 Year Treasury",
    "SHY": "iShares 1-3 Year Treasury",
    "BNDX": "Vanguard Intl Bond",
    "TIP": "iShares TIPS Bond",
    # Cash proxy / benchmark risk-free
    "BIL": "SPDR 1-3 Month T-Bill",
}

CASH_PROXY = "BIL"
TOP_N = 3
INITIAL_CAPITAL = 100.0

# Momentum lookback periods (in mesi)
LOOKBACK_PERIODS = [3, 6, 12]
SKIP_MONTH = 1  # Escludi il mese più recente (Novy-Marx 2012)

# Volatility
VOL_LOOKBACK_DAYS = 60  # Giorni per calcolo volatilità
VOL_TARGET = 0.10  # Target volatilità annualizzata (10%)
TRADING_DAYS_PER_YEAR = 252

# Position constraints
MAX_POSITION_WEIGHT = 0.50  # Max 50% per posizione
MIN_POSITION_WEIGHT = 0.10  # Min 10% per posizione

# Circuit breaker thresholds
CB_LEVEL_1_DRAWDOWN = -0.15  # -15% → riduzione al 50%
CB_LEVEL_2_DRAWDOWN = -0.25  # -25% → exit totale a cash
POSITION_STOP_LOSS = -0.12   # -12% → vendita singola posizione

PORTFOLIO_PATH = Path(__file__).parent / "portfolio.json"


# ─────────────────────────────────────────────────────────────────────
# Market Data Provider
# ─────────────────────────────────────────────────────────────────────

class MarketDataProvider:
    """Scarica e processa dati di mercato da Yahoo Finance."""

    def __init__(self, tickers: list[str], lookback_months: int = 14):
        self.tickers = tickers
        self.lookback_months = lookback_months
        self._prices: pd.DataFrame | None = None
        self._returns: pd.DataFrame | None = None
        self._is_mocked = False

    def set_data_slice(self, prices: pd.DataFrame, returns: pd.DataFrame) -> None:
        """Inietta dati storici troncati (usato per il backtesting)."""
        self._prices = prices
        self._returns = returns
        self._is_mocked = True

    def fetch(self) -> None:
        """Scarica i prezzi di chiusura aggiustati per tutti i ticker."""
        if self._is_mocked:
            return  # Bypass se i dati sono stati iniettati
        import os
        # Workaround: disabilita il cookie consent check di yfinance
        # che può fallire in ambienti con restrizioni di rete
        os.environ["YF_CONSENT"] = "1"

        period = f"{self.lookback_months}mo"
        print(f"\n📊 Scaricamento dati di mercato ({period})...")

        # Retry con backoff per gestire rate limiting
        import time
        max_retries = 3
        data = pd.DataFrame()

        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers=self.tickers,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False,  # Evita problemi di concorrenza
                )
                if not data.empty:
                    break
            except Exception as e:
                print(f"  ⚠️  Tentativo {attempt + 1}/{max_retries} fallito: {e}")
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"  ⏳ Riprovo tra {wait} secondi...")
                    time.sleep(wait)

        if data.empty:
            raise RuntimeError("Nessun dato ricevuto da Yahoo Finance.")

        # Estrai i prezzi di chiusura
        if isinstance(data.columns, pd.MultiIndex):
            self._prices = data["Close"]
        else:
            # Singolo ticker
            self._prices = data[["Close"]]
            self._prices.columns = self.tickers

        # Rimuovi colonne con troppi NaN
        self._prices = self._prices.dropna(axis=1, thresh=int(len(self._prices) * 0.8))
        self._returns = self._prices.pct_change().dropna()

        available = list(self._prices.columns)
        missing = set(self.tickers) - set(available)
        if missing:
            print(f"  ⚠️  Ticker non disponibili (rimossi): {', '.join(missing)}")
        print(f"  ✅ {len(available)} ticker caricati, {len(self._prices)} giorni di dati")

    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            raise RuntimeError("Dati non ancora caricati. Chiamare fetch() prima.")
        return self._prices

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns is None:
            raise RuntimeError("Dati non ancora caricati. Chiamare fetch() prima.")
        return self._returns

    def get_current_prices(self) -> pd.Series:
        """Ritorna l'ultimo prezzo di chiusura per ogni ticker."""
        return self.prices.iloc[-1]

    def get_period_return(self, months: int, skip: int = 0) -> pd.Series:
        """Calcola il rendimento su N mesi, saltando gli ultimi `skip` mesi."""
        trading_days_per_month = TRADING_DAYS_PER_YEAR / 12
        end_offset = int(skip * trading_days_per_month)
        start_offset = int((months + skip) * trading_days_per_month)

        if start_offset > len(self.prices):
            # Non abbastanza dati, usa tutto quello disponibile
            start_offset = len(self.prices) - 1

        end_idx = len(self.prices) - 1 - end_offset if end_offset > 0 else len(self.prices) - 1
        start_idx = len(self.prices) - 1 - start_offset

        if start_idx < 0:
            start_idx = 0

        start_prices = self.prices.iloc[start_idx]
        end_prices = self.prices.iloc[end_idx]

        return (end_prices / start_prices) - 1

    def get_volatility(self, days: int = VOL_LOOKBACK_DAYS) -> pd.Series:
        """Calcola la volatilità annualizzata sugli ultimi N giorni."""
        recent_returns = self.returns.tail(days)
        daily_vol = recent_returns.std()
        return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)


# ─────────────────────────────────────────────────────────────────────
# Momentum Scorer
# ─────────────────────────────────────────────────────────────────────

class MomentumScorer:
    """Calcola il momentum score blended multi-periodo, aggiustato per volatilità."""

    def __init__(self, data: MarketDataProvider):
        self.data = data

    def compute_scores(self) -> pd.DataFrame:
        """
        Calcola il momentum score per ogni asset.

        Returns:
            DataFrame con colonne: ticker, raw_score, volatility, adj_score, rank,
                                   ret_3m, ret_6m, ret_12m
        """
        results = []

        for ticker in self.data.prices.columns:
            period_returns = {}
            for months in LOOKBACK_PERIODS:
                try:
                    ret = self.data.get_period_return(months, skip=SKIP_MONTH)
                    period_returns[months] = ret[ticker]
                except (KeyError, IndexError):
                    period_returns[months] = 0.0

            # Blended momentum score (media equi-ponderata)
            raw_score = np.mean(list(period_returns.values()))

            # Volatilità annualizzata
            vol = self.data.get_volatility()[ticker]

            # Volatility-adjusted score
            adj_score = raw_score / vol if vol > 0.001 else raw_score

            results.append({
                "ticker": ticker,
                "name": UNIVERSE.get(ticker, ticker),
                "ret_3m": period_returns.get(3, 0.0),
                "ret_6m": period_returns.get(6, 0.0),
                "ret_12m": period_returns.get(12, 0.0),
                "raw_score": raw_score,
                "volatility": vol,
                "adj_score": adj_score,
            })

        df = pd.DataFrame(results)
        df = df.sort_values("adj_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df


# ─────────────────────────────────────────────────────────────────────
# Portfolio Allocator
# ─────────────────────────────────────────────────────────────────────

class PortfolioAllocator:
    """Seleziona asset e calcola i pesi ottimali del portafoglio."""

    def __init__(self, data: MarketDataProvider, scores: pd.DataFrame):
        self.data = data
        self.scores = scores

    def allocate(self) -> dict[str, float]:
        """
        Esegue la pipeline di allocazione completa:
        1. Selezione Top-N
        2. Filtro Momentum Assoluto
        3. Inverse-Volatility Weighting
        4. Volatility Targeting
        5. Vincoli di peso

        Returns:
            Dict {ticker: weight} con i pesi target
        """
        # 1. Selezione Top-N
        top_n = self.scores.head(TOP_N).copy()
        print(f"\n🏆 Top {TOP_N} per momentum score (vol-adjusted):")
        # Trova il rendimento di BIL per il filtro assoluto
        bil_ret = 0.0
        if CASH_PROXY in self.scores["ticker"].values:
            bil_row = self.scores[self.scores["ticker"] == CASH_PROXY]
            if not bil_row.empty:
                bil_ret = bil_row.iloc[0]["ret_12m"]

        # Escludi CASH_PROXY dalla selezione degli asset investibili
        investable_scores = self.scores[self.scores["ticker"] != CASH_PROXY]

        # 1. Selezione Top-N
        top_n = investable_scores.head(TOP_N).copy()
        print(f"\n🏆 Top {TOP_N} per momentum score (vol-adjusted):")
        for _, row in top_n.iterrows():
            print(f"   #{int(row['rank'])} {row['ticker']:5s} | Score: {row['adj_score']:+.4f} | "
                  f"3m: {row['ret_3m']:+.1%} | 6m: {row['ret_6m']:+.1%} | 12m: {row['ret_12m']:+.1%}")

        # 2. Filtro Momentum Assoluto
        selected = {}
        for _, row in top_n.iterrows():
            ticker = row["ticker"]
            if row["ret_12m"] > bil_ret:
                selected[ticker] = row["volatility"]
                print(f"   ✅ {ticker} passa il filtro assoluto (12m: {row['ret_12m']:+.1%} > BIL: {bil_ret:+.1%})")
            else:
                print(f"   ❌ {ticker} FALLISCE il filtro assoluto (12m: {row['ret_12m']:+.1%} ≤ BIL: {bil_ret:+.1%}) → Cash")

        if not selected:
            print("\n   🛡️ Tutti gli asset hanno momentum negativo → 100% Cash")
            return {CASH_PROXY: 1.0}

        # 3. Inverse-Volatility Weighting
        weights = self._inverse_vol_weights(selected)

        # 4. Applica vincoli di peso
        weights = self._apply_constraints(weights)

        # 5. Volatility Targeting
        weights = self._apply_vol_target(weights)

        return weights

    def _inverse_vol_weights(self, selected: dict[str, float]) -> dict[str, float]:
        """Calcola i pesi inversamente proporzionali alla volatilità."""
        inv_vols = {ticker: 1.0 / vol for ticker, vol in selected.items() if vol > 0.001}

        if not inv_vols:
            # Fallback: equal weight
            n = len(selected)
            return {ticker: 1.0 / n for ticker in selected}

        total_inv_vol = sum(inv_vols.values())
        weights = {ticker: iv / total_inv_vol for ticker, iv in inv_vols.items()}

        print(f"\n⚖️  Pesi Inverse-Volatility:")
        for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
            vol = selected[ticker]
            print(f"   {ticker:5s} → {w:.1%} (vol: {vol:.1%})")

        return weights

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Applica vincoli min/max ai pesi, ridistribuendo l'eccesso."""
        constrained = dict(weights)
        iterations = 0
        max_iterations = 10

        while iterations < max_iterations:
            excess = 0.0
            below_min = []

            for ticker, w in constrained.items():
                if w > MAX_POSITION_WEIGHT:
                    excess += w - MAX_POSITION_WEIGHT
                    constrained[ticker] = MAX_POSITION_WEIGHT
                elif w < MIN_POSITION_WEIGHT:
                    below_min.append(ticker)

            if excess == 0 and not below_min:
                break

            # Ridistribuisci l'eccesso
            eligible = [t for t in constrained if constrained[t] < MAX_POSITION_WEIGHT and t not in below_min]
            if eligible and excess > 0:
                per_ticker = excess / len(eligible)
                for t in eligible:
                    constrained[t] += per_ticker

            # Gestisci posizioni sotto il minimo
            for t in below_min:
                constrained[t] = MIN_POSITION_WEIGHT

            # Rinormalizza
            total = sum(constrained.values())
            if total > 0:
                constrained = {t: w / total for t, w in constrained.items()}

            iterations += 1

        return constrained

    def _apply_vol_target(self, weights: dict[str, float]) -> dict[str, float]:
        """Scala i pesi per raggiungere la volatilità target."""
        # Stima la volatilità del portafoglio (semplificata: media ponderata)
        portfolio_vol = sum(
            w * self.data.get_volatility().get(ticker, 0.10)
            for ticker, w in weights.items()
        )

        if portfolio_vol <= 0.001:
            return weights

        leverage = min(VOL_TARGET / portfolio_vol, 1.0)

        if leverage < 1.0:
            cash_weight = 1.0 - leverage
            scaled = {ticker: w * leverage for ticker, w in weights.items()}
            scaled[CASH_PROXY] = scaled.get(CASH_PROXY, 0.0) + cash_weight

            print(f"\n🎯 Volatility Targeting:")
            print(f"   Portfolio vol: {portfolio_vol:.1%} | Target: {VOL_TARGET:.1%} | Scala: {leverage:.2f}")
            print(f"   Cash aggiunto: {cash_weight:.1%}")

            return scaled
        else:
            print(f"\n🎯 Volatility Targeting:")
            print(f"   Portfolio vol: {portfolio_vol:.1%} ≤ Target: {VOL_TARGET:.1%} | Nessun aggiustamento")
            return weights


# ─────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────

class RiskManager:
    """Monitora e gestisce il rischio del portafoglio."""

    def __init__(self, portfolio: dict[str, Any]):
        self.portfolio = portfolio

    def check_portfolio_drawdown(self, current_value: float) -> str:
        """
        Verifica il drawdown del portafoglio dal picco.

        Returns:
            "normal" | "level_1" | "level_2"
        """
        peak = self.portfolio.get("peak_value", current_value)
        if current_value > peak:
            self.portfolio["peak_value"] = current_value
            peak = current_value

        if peak <= 0:
            return "normal"

        drawdown = (current_value - peak) / peak

        if drawdown <= CB_LEVEL_2_DRAWDOWN:
            print(f"\n🛑 CIRCUIT BREAKER LIVELLO 2: Drawdown {drawdown:.1%} (soglia: {CB_LEVEL_2_DRAWDOWN:.0%})")
            print(f"   → EXIT TOTALE A CASH")
            return "level_2"
        elif drawdown <= CB_LEVEL_1_DRAWDOWN:
            print(f"\n⚠️  CIRCUIT BREAKER LIVELLO 1: Drawdown {drawdown:.1%} (soglia: {CB_LEVEL_1_DRAWDOWN:.0%})")
            print(f"   → Riduzione esposizione al 50%")
            return "level_1"
        else:
            if drawdown < 0:
                print(f"\n📉 Drawdown corrente: {drawdown:.1%} (sotto soglia, nessuna azione)")
            return "normal"

    def check_position_stops(
        self, holdings: list[dict], current_prices: pd.Series
    ) -> list[str]:
        """
        Verifica i trailing stop per le singole posizioni.

        Returns:
            Lista di ticker da vendere per stop-loss.
        """
        to_sell = []
        for holding in holdings:
            ticker = holding["ticker"]
            buy_price = holding["buy_price"]
            if ticker not in current_prices.index:
                continue
            current_price = current_prices[ticker]
            pnl = (current_price - buy_price) / buy_price

            if pnl <= POSITION_STOP_LOSS:
                print(f"   🛑 STOP-LOSS: {ticker} a {pnl:+.1%} (soglia: {POSITION_STOP_LOSS:.0%})")
                to_sell.append(ticker)

        return to_sell

    def apply_circuit_breaker(
        self, weights: dict[str, float], cb_level: str
    ) -> dict[str, float]:
        """Modifica i pesi in base al livello del circuit breaker."""
        if cb_level == "level_2":
            return {CASH_PROXY: 1.0}
        elif cb_level == "level_1":
            adjusted = {}
            for ticker, w in weights.items():
                if ticker == CASH_PROXY:
                    adjusted[ticker] = w
                else:
                    adjusted[ticker] = w * 0.5
            cash_freed = sum(w * 0.5 for t, w in weights.items() if t != CASH_PROXY)
            adjusted[CASH_PROXY] = adjusted.get(CASH_PROXY, 0.0) + cash_freed
            return adjusted
        return weights


# ─────────────────────────────────────────────────────────────────────
# Trading Engine
# ─────────────────────────────────────────────────────────────────────

class TradingEngine:
    """Orchestratore principale: esegue il ciclo di ribilanciamento."""

    def __init__(self):
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self) -> dict[str, Any]:
        """Carica lo stato del portafoglio da file o ne crea uno nuovo."""
        if PORTFOLIO_PATH.exists():
            with open(PORTFOLIO_PATH) as f:
                portfolio = json.load(f)
            print(f"📂 Portafoglio caricato: ${portfolio['cash']:.2f} cash, "
                  f"{len(portfolio.get('holdings', []))} posizioni")
            return portfolio

        print(f"🆕 Creazione nuovo portafoglio con ${INITIAL_CAPITAL:.2f}")
        portfolio = {
            "cash": INITIAL_CAPITAL,
            "holdings": [],
            "peak_value": INITIAL_CAPITAL,
            "trade_history": [],
            "value_history": [
                {
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "value": INITIAL_CAPITAL,
                    "cash": INITIAL_CAPITAL,
                    "invested": 0.0,
                }
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return portfolio

    def _save_portfolio(self) -> None:
        """Salva lo stato del portafoglio su file."""
        PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PORTFOLIO_PATH, "w") as f:
            json.dump(self.portfolio, f, indent=2, default=str)
        print(f"\n💾 Portafoglio salvato in {PORTFOLIO_PATH}")

    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """Calcola il valore totale del portafoglio."""
        total = self.portfolio["cash"]
        for holding in self.portfolio.get("holdings", []):
            ticker = holding["ticker"]
            if ticker in current_prices.index:
                total += holding["quantity"] * current_prices[ticker]
        return total

    def _execute_sell(self, ticker: str, quantity: float, price: float, reason: str) -> None:
        """Esegue una vendita simulata."""
        proceeds = quantity * price
        self.portfolio["cash"] += proceeds

        # Rimuovi dalla lista holdings
        self.portfolio["holdings"] = [
            h for h in self.portfolio["holdings"] if h["ticker"] != ticker
        ]

        trade = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "action": "SELL",
            "ticker": ticker,
            "quantity": round(quantity, 6),
            "price": round(price, 2),
            "value": round(proceeds, 2),
            "reason": reason,
        }
        self.portfolio["trade_history"].append(trade)
        print(f"   📤 VENDITA: {quantity:.4f} × {ticker} @ ${price:.2f} = ${proceeds:.2f} ({reason})")

    def _execute_buy(self, ticker: str, amount: float, price: float, reason: str) -> None:
        """Esegue un acquisto simulato (quote frazionarie)."""
        if amount <= 0.01 or price <= 0:
            return

        quantity = amount / price
        self.portfolio["cash"] -= amount

        # Aggiungi o aggiorna holding
        existing = [h for h in self.portfolio["holdings"] if h["ticker"] == ticker]
        if existing:
            h = existing[0]
            total_qty = h["quantity"] + quantity
            total_cost = h["buy_price"] * h["quantity"] + amount
            h["quantity"] = total_qty
            h["buy_price"] = total_cost / total_qty  # Prezzo medio
        else:
            self.portfolio["holdings"].append({
                "ticker": ticker,
                "quantity": round(quantity, 6),
                "buy_price": round(price, 2),
                "buy_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            })

        trade = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "action": "BUY",
            "ticker": ticker,
            "quantity": round(quantity, 6),
            "price": round(price, 2),
            "value": round(amount, 2),
            "reason": reason,
        }
        self.portfolio["trade_history"].append(trade)
        print(f"   📥 ACQUISTO: {quantity:.4f} × {ticker} @ ${price:.2f} = ${amount:.2f} ({reason})")

    def _rebalance(self, target_weights: dict[str, float], current_prices: pd.Series) -> None:
        """Ribilancia il portafoglio verso i pesi target."""
        portfolio_value = self._calculate_portfolio_value(current_prices)

        print(f"\n🔄 Ribilanciamento (valore portafoglio: ${portfolio_value:.2f}):")

        # 1. Vendi tutto ciò che non è nei target
        current_tickers = {h["ticker"] for h in self.portfolio["holdings"]}
        target_tickers = {t for t, w in target_weights.items() if w > 0.01 and t != CASH_PROXY}

        for holding in list(self.portfolio["holdings"]):
            ticker = holding["ticker"]
            if ticker not in target_tickers:
                if ticker in current_prices.index:
                    self._execute_sell(ticker, holding["quantity"], current_prices[ticker], "Non più in Top-N")

        # 2. Calcola allocazione target in dollari
        target_values = {
            ticker: portfolio_value * weight
            for ticker, weight in target_weights.items()
            if ticker != CASH_PROXY and weight > 0.01
        }

        # 3. Vendi eccesso e compra mancanze per le posizioni target
        for holding in list(self.portfolio["holdings"]):
            ticker = holding["ticker"]
            if ticker in target_values and ticker in current_prices.index:
                current_val = holding["quantity"] * current_prices[ticker]
                target_val = target_values[ticker]
                diff = target_val - current_val

                if diff < -1.0:  # Vendi eccesso (soglia $1 per evitare micro-trade)
                    qty_to_sell = abs(diff) / current_prices[ticker]
                    self._execute_sell(ticker, qty_to_sell, current_prices[ticker], "Ribilanciamento (riduzione)")
                    target_values[ticker] = 0  # Già gestito
                elif diff > 1.0:
                    target_values[ticker] = diff  # Compra solo la differenza
                else:
                    target_values[ticker] = 0  # Differenza trascurabile

        # 4. Compra nuove posizioni
        for ticker, amount in target_values.items():
            if amount > 1.0 and ticker in current_prices.index:
                buy_amount = min(amount, self.portfolio["cash"])
                if buy_amount > 0.50:
                    self._execute_buy(ticker, buy_amount, current_prices[ticker], "Ribilanciamento")

    def run(self) -> None:
        """Esegue un ciclo completo di ribilanciamento."""
        print("=" * 65)
        print("  AEGIS MOMENTUM — Weekly Rebalance")
        print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 65)

        # 1. Scarica dati di mercato
        tickers = list(UNIVERSE.keys())
        data = MarketDataProvider(tickers, lookback_months=14)
        data.fetch()

        current_prices = data.get_current_prices()

        # 2. Calcola valore corrente e verifica circuit breaker
        portfolio_value = self._calculate_portfolio_value(current_prices)
        risk_mgr = RiskManager(self.portfolio)

        # Verifica stop-loss singole posizioni
        stops = risk_mgr.check_position_stops(self.portfolio.get("holdings", []), current_prices)
        for ticker in stops:
            holding = next((h for h in self.portfolio["holdings"] if h["ticker"] == ticker), None)
            if holding and ticker in current_prices.index:
                self._execute_sell(ticker, holding["quantity"], current_prices[ticker], "Stop-loss")

        # Ricalcola valore dopo eventuali stop-loss
        portfolio_value = self._calculate_portfolio_value(current_prices)
        cb_level = risk_mgr.check_portfolio_drawdown(portfolio_value)

        # 3. Calcola momentum scores
        scorer = MomentumScorer(data)
        scores = scorer.compute_scores()

        print(f"\n📊 Classifica Momentum (tutti i {len(scores)} ETF):")
        print(f"   {'#':>3} {'Ticker':6s} {'3m':>8s} {'6m':>8s} {'12m':>8s} {'Score':>8s} {'Vol':>7s}")
        print(f"   {'─'*3} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
        for _, row in scores.iterrows():
            marker = " 🏅" if row["rank"] <= TOP_N else ""
            print(f"   {int(row['rank']):3d} {row['ticker']:6s} "
                  f"{row['ret_3m']:+7.1%} {row['ret_6m']:+7.1%} {row['ret_12m']:+7.1%} "
                  f"{row['adj_score']:+7.3f} {row['volatility']:6.1%}{marker}")

        # 4. Calcola allocazione
        allocator = PortfolioAllocator(data, scores)
        target_weights = allocator.allocate()

        # 5. Applica circuit breaker
        target_weights = risk_mgr.apply_circuit_breaker(target_weights, cb_level)

        # 6. Esegui ribilanciamento
        self._rebalance(target_weights, current_prices)

        # 7. Aggiorna valore e salva
        final_value = self._calculate_portfolio_value(current_prices)
        if final_value > self.portfolio.get("peak_value", 0):
            self.portfolio["peak_value"] = final_value

        # Registra nella value history
        invested_value = sum(
            h["quantity"] * current_prices.get(h["ticker"], 0)
            for h in self.portfolio["holdings"]
            if h["ticker"] in current_prices.index
        )
        self.portfolio.setdefault("value_history", []).append({
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "value": round(final_value, 2),
            "cash": round(self.portfolio["cash"], 2),
            "invested": round(invested_value, 2),
        })

        # Salva scores nel portafoglio per la dashboard
        self.portfolio["last_scores"] = scores.to_dict(orient="records")
        self.portfolio["last_run"] = datetime.now(timezone.utc).isoformat()
        self.portfolio["target_weights"] = {
            ticker: round(w, 4) for ticker, w in target_weights.items()
        }

        self._save_portfolio()

        # 8. Report finale
        self._print_report(final_value, target_weights, current_prices)

    def _print_report(
        self, value: float, weights: dict[str, float], prices: pd.Series
    ) -> None:
        """Stampa il report finale del ribilanciamento."""
        initial = INITIAL_CAPITAL
        total_return = (value - initial) / initial
        days_active = 1
        created = self.portfolio.get("created_at")
        if created:
            try:
                start = datetime.fromisoformat(created)
                days_active = max((datetime.now(timezone.utc) - start).days, 1)
            except (ValueError, TypeError):
                pass

        ann_return = (1 + total_return) ** (365 / days_active) - 1 if days_active > 1 else 0
        peak = self.portfolio.get("peak_value", value)
        max_dd = (value - peak) / peak if peak > 0 else 0

        print("\n" + "=" * 65)
        print("  📋 REPORT FINALE")
        print("=" * 65)
        print(f"  💰 Valore portafoglio:  ${value:.2f}")
        print(f"  💵 Cash disponibile:    ${self.portfolio['cash']:.2f}")
        print(f"  📈 Rendimento totale:   {total_return:+.2%}")
        print(f"  📊 Rendimento ann.:     {ann_return:+.2%}")
        print(f"  📉 Max Drawdown:        {max_dd:+.2%}")
        print(f"  🏔️  Picco storico:       ${peak:.2f}")
        print(f"  📅 Giorni attivi:       {days_active}")

        if self.portfolio.get("holdings"):
            print(f"\n  📦 Posizioni attuali:")
            for h in self.portfolio["holdings"]:
                ticker = h["ticker"]
                qty = h["quantity"]
                buy_px = h["buy_price"]
                cur_px = prices.get(ticker, buy_px)
                val = qty * cur_px
                pnl = (cur_px - buy_px) / buy_px if buy_px > 0 else 0
                print(f"     {ticker:6s} {qty:.4f} × ${cur_px:.2f} = ${val:.2f} ({pnl:+.1%})")

        print(f"\n  🎯 Allocazione target:")
        for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
            if w > 0.005:
                name = UNIVERSE.get(ticker, ticker)
                print(f"     {ticker:6s} {w:5.1%}  {name}")

        print("=" * 65)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        engine = TradingEngine()
        engine.run()
    except Exception as e:
        print(f"\n❌ Errore: {e}", file=sys.stderr)
        raise
