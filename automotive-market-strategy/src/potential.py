"""
EV market potential for Company X in 2030.

market scenario (slow / base / fast EV adoption)  x  share scenario (how much of the EV
segment Company X keeps)  ->  units and revenue.
"""
import numpy as np
import pandas as pd

from src.data import COMPANY


def share_scenarios(overview: pd.DataFrame, target_year=2030, floor=25.0) -> dict:
    """Company X share of the EV segment in target_year under three assumptions."""
    share = overview["company_share_of_ev_segment_pct"]
    last_year = int(share.index.max())
    years_ahead = target_year - last_year
    recent = share.loc[last_year - 3:]                       # last 4 years
    slope = np.polyfit(recent.index, recent.values, 1)[0]    # pp per year (negative)
    return {
        "defend (hold share)": share.iloc[-1],
        "slow erosion (-1.5 pp/yr)": max(share.iloc[-1] - 1.5 * years_ahead, floor),
        f"trend continues ({slope:.1f} pp/yr)": max(share.iloc[-1] + slope * years_ahead, floor),
    }


def company_ev_price(models: pd.DataFrame, year: int) -> float:
    ev = models[(models["fuel_type"] == "EV") & (models["year"] == year)]
    return float((ev["units"] * ev["avg_price_usd"]).sum() / ev["units"].sum())


def potential_table(forecast: pd.DataFrame, overview: pd.DataFrame, models: pd.DataFrame,
                    target_year=2030) -> pd.DataFrame:
    last_year = int(overview.index.max())
    price = company_ev_price(models, last_year)
    shares = share_scenarios(overview, target_year)
    current_units = overview.loc[last_year, "company_ev_units"]
    rows = []
    for _, f in forecast[forecast["year"] == target_year].iterrows():
        for share_name, share in shares.items():
            units = f["ev_units"] * share / 100
            rows.append({
                "ev_adoption": f["scenario"],
                "company_share_scenario": share_name,
                "ev_market_units": int(f["ev_units"]),
                "company_ev_share_pct": round(share, 1),
                "company_ev_units": int(units),
                "growth_vs_now_x": round(units / current_units, 1),
                "revenue_busd": round(units * price / 1e9, 2),
            })
    out = pd.DataFrame(rows)
    out.attrs.update(price=price, base_year=last_year, company=COMPANY)
    return out
