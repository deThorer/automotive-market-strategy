"""Descriptive market analysis: fuel mix, EV segment shares, company and model performance."""
import pandas as pd

from src.data import COMPANY


def yearly_fuel_mix(market: pd.DataFrame) -> pd.DataFrame:
    """Units and % share of each fuel type per year (whole country)."""
    units = market.pivot_table(index="year", columns="fuel_type", values="units", aggfunc="sum")
    share = units.div(units.sum(axis=1), axis=0) * 100
    return units, share.round(2)


def growth_by_fuel(units: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """YoY growth for the last year and CAGR between two years."""
    years = end - start
    cagr = (units.loc[end] / units.loc[start]) ** (1 / years) - 1
    yoy = units.loc[end] / units.loc[end - 1] - 1
    return pd.DataFrame({"cagr_pct": cagr * 100, "last_yoy_pct": yoy * 100}).round(1)


def ev_segment_shares(market: pd.DataFrame) -> pd.DataFrame:
    """% of the EV segment held by each brand, per year."""
    ev = market[market["fuel_type"] == "EV"]
    units = ev.pivot_table(index="year", columns="brand", values="units", aggfunc="sum").fillna(0)
    return (units.div(units.sum(axis=1), axis=0) * 100).round(1)


def company_overview(market: pd.DataFrame) -> pd.DataFrame:
    total = market.groupby("year")["units"].sum()
    ev_total = market[market["fuel_type"] == "EV"].groupby("year")["units"].sum()
    cx = market[market["brand"] == COMPANY]
    cx_total = cx.groupby("year")["units"].sum()
    cx_ev = cx[cx["fuel_type"] == "EV"].groupby("year")["units"].sum()
    return pd.DataFrame({
        "market_units": total,
        "company_units": cx_total,
        "company_market_share_pct": 100 * cx_total / total,
        "ev_market_units": ev_total,
        "company_ev_units": cx_ev,
        "company_share_of_ev_segment_pct": 100 * cx_ev / ev_total,
        "ev_pct_of_company_sales": 100 * cx_ev / cx_total,
    }).round(2)


def model_performance(models: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-model volume, growth, revenue and share of Company X EV sales for one year."""
    m = models.assign(revenue=models["units"] * models["avg_price_usd"])
    yearly = m.groupby(["model", "fuel_type", "year"]).agg(units=("units", "sum"),
                                                          revenue=("revenue", "sum")).reset_index()
    cur = yearly[yearly["year"] == year].set_index("model")
    prev = yearly[yearly["year"] == year - 1].set_index("model")
    out = cur[["fuel_type", "units"]].copy()
    out["yoy_pct"] = (100 * (cur["units"] / prev["units"] - 1)).round(1)
    out["revenue_musd"] = (cur["revenue"] / 1e6).round(1)
    out["avg_price_usd"] = (cur["revenue"] / cur["units"]).round(0)
    ev_units = out.loc[out["fuel_type"] == "EV", "units"].sum()
    out["share_of_company_ev_pct"] = (100 * out["units"] / ev_units).where(out["fuel_type"] == "EV").round(1)
    return out.sort_values("units", ascending=False)


def ev_model_mix(models: pd.DataFrame) -> pd.DataFrame:
    """Share of each EV model in Company X EV sales, per year."""
    ev = models[models["fuel_type"] == "EV"]
    units = ev.pivot_table(index="year", columns="model", values="units", aggfunc="sum").fillna(0)
    return (units.div(units.sum(axis=1), axis=0) * 100).round(1)


def review_summary(reviews: pd.DataFrame) -> pd.DataFrame:
    return reviews.pivot(index="model", columns="aspect", values="score")
