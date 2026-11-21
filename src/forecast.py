"""
Forecasting EV vs petrol (gasoline) demand to 2030.

EV share follows an S-curve (logistic):     share(t) = L / (1 + exp(-k (t - t0)))
  L  = saturation level (the share EVs eventually reach)
  k  = adoption speed
  t0 = midpoint year

Problem: the market is still at the bottom of the curve (~2% EV). In that region the
curve looks exponential and the data can't tell what L is. So instead of fitting L,
we FIX it for three scenarios and fit k and t0 for each one.

With L fixed the fit is just linear regression, because
    ln( share / (L - share) ) = k * t - k * t0
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCENARIOS = {"slow": 0.30, "base": 0.50, "fast": 0.70}   # EV saturation share


@dataclass
class Logistic:
    L: float
    k: float
    t0: float

    def predict(self, t):
        return self.L / (1 + np.exp(-self.k * (np.asarray(t) - self.t0)))


def to_decimal_year(months: pd.Series) -> np.ndarray:
    return (months.dt.year + (months.dt.month - 1) / 12).to_numpy()


def monthly_ev_share(market: pd.DataFrame) -> pd.DataFrame:
    total = market.groupby("month")["units"].sum()
    ev = market[market["fuel_type"] == "EV"].groupby("month")["units"].sum()
    out = (ev / total).rename("ev_share").reset_index()
    out["t"] = to_decimal_year(out["month"])
    return out


def fit_logistic(t, share, L) -> Logistic:
    if np.any(share >= L):
        raise ValueError(f"observed share already above saturation L={L}")
    y = np.log(share / (L - share))
    k, intercept = np.polyfit(t, y, 1)
    return Logistic(L=L, k=k, t0=-intercept / k)


def fit_exponential(t, share):
    b, a = np.polyfit(t, np.log(share), 1)
    return lambda x: np.exp(a + b * np.asarray(x))


def fit_linear(t, share):
    b, a = np.polyfit(t, share, 1)
    return lambda x: a + b * np.asarray(x)


def mape(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def backtest(ev: pd.DataFrame, train_end="2022-12-01") -> pd.DataFrame:
    """Fit on data up to train_end, score on the rest (MAPE of monthly EV share)."""
    train = ev[ev["month"] <= train_end]
    test = ev[ev["month"] > train_end]
    rows = [("linear trend", fit_linear(train.t, train.ev_share)),
            ("exponential", fit_exponential(train.t, train.ev_share))]
    rows += [(f"logistic L={L:.0%}", fit_logistic(train.t, train.ev_share, L).predict)
             for L in SCENARIOS.values()]
    return pd.DataFrame([{"model": name, "test_mape_pct": round(mape(test.ev_share, f(test.t)), 1)}
                         for name, f in rows]).sort_values("test_mape_pct")


def total_market_forecast(market: pd.DataFrame, years) -> pd.Series:
    """Log-linear trend on yearly totals. 2020 is skipped (COVID lockdown dip)."""
    yearly = market.groupby("year")["units"].sum()
    fit = yearly.drop(2020, errors="ignore")
    b, a = np.polyfit(fit.index, np.log(fit.values), 1)
    return pd.Series(np.exp(a + b * np.asarray(years)), index=years, name="total_units"), np.exp(b) - 1


def other_fuel_shares(market: pd.DataFrame, years) -> pd.DataFrame:
    """Linear trend for Diesel / CNG / Hybrid shares (clipped at 1%). Petrol = what is left."""
    units = market.pivot_table(index="year", columns="fuel_type", values="units", aggfunc="sum")
    share = units.div(units.sum(axis=1), axis=0)
    out = {}
    for fuel in ["Diesel", "CNG", "Hybrid"]:
        b, a = np.polyfit(share.index, share[fuel], 1)
        out[fuel] = np.clip(a + b * np.asarray(years), 0.01, None)
    return pd.DataFrame(out, index=years)


def scenario_forecast(market: pd.DataFrame, years=range(2025, 2031)) -> pd.DataFrame:
    years = list(years)
    ev = monthly_ev_share(market)
    total, growth = total_market_forecast(market, years)
    others = other_fuel_shares(market, years)
    rows = []
    for name, L in SCENARIOS.items():
        model = fit_logistic(ev.t, ev.ev_share, L)
        ev_share = model.predict(np.array(years) + 0.5)          # mid-year
        other = others.sum(axis=1).to_numpy()
        petrol_share = np.clip(1 - ev_share - other, 0, None)
        for i, y in enumerate(years):
            rows.append({"scenario": name, "year": y, "saturation_L": L, "k": round(model.k, 3),
                         "total_units": round(total[y]), "ev_share_pct": round(100 * ev_share[i], 2),
                         "ev_units": round(total[y] * ev_share[i]),
                         "petrol_share_pct": round(100 * petrol_share[i], 2),
                         "petrol_units": round(total[y] * petrol_share[i])})
    df = pd.DataFrame(rows)
    df.attrs["market_growth"] = growth
    return df
