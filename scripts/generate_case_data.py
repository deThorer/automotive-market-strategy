"""
Generate the SIMULATED case-study dataset used in this project.

The case: an emerging car market where electric vehicles (EVs) are still a small share
of sales but growing fast. "Company X" is a mainstream maker that moved into EVs early
and holds roughly 73% of the country's EV segment, while most of its own volume is
still petrol/diesel/CNG. Competitors are launching EVs and the lead is shrinking.

All brands, models and numbers are made up. The shape of the market (early-stage
S-curve EV adoption, diesel declining, festive-season peak in Oct/Nov) is modelled on
real emerging markets so the methods are realistic.

Output (data/raw/):
    market_sales.csv   month, brand, fuel_type, units          (whole country)
    company_models.csv month, model, segment, fuel_type, units, avg_price_usd
    model_reviews.csv  model, aspect, score, n_reviews         (1-5 customer scores)

    python scripts/generate_case_data.py [--seed 3]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw"

MONTHS = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
FUELS = ["EV", "Petrol", "Diesel", "CNG", "Hybrid"]

# competitor -> month it enters the EV segment, relative EV strength once established
EV_COMPETITORS = {"Competitor A": ("2020-01", 1.0), "Competitor B": ("2018-01", 0.35),
                  "Competitor C": ("2023-01", 0.9), "Others": ("2018-01", 0.25)}
ICE_COMPETITORS = ["Competitor A", "Competitor B", "Competitor C", "Competitor D", "Others"]

# Company X models: (model, segment, fuel_type, launch, base_price_usd, weight within fuel)
MODELS = [
    ("X-City EV",      "Compact SUV", "EV",     "2018-01", 17500, None),   # weight set below
    ("X-Range EV",     "Mid SUV",     "EV",     "2020-06", 23000, None),
    ("X-Pro EV",       "Premium SUV", "EV",     "2023-04", 29000, None),
    ("X-City Petrol",  "Compact SUV", "Petrol", "2018-01", 10500, 0.55),
    ("X-Trail Petrol", "Mid SUV",     "Petrol", "2018-01", 16000, 0.45),
    ("X-Trail Diesel", "Mid SUV",     "Diesel", "2018-01", 17500, 1.0),
    ("X-City CNG",     "Compact SUV", "CNG",    "2018-01", 11500, 1.0),
]


def t_years(months):
    return (months.year - 2018) + (months.month - 1) / 12


def ramp(t, start, years=1.5):
    """0 before launch, rises to 1 over `years` after launch (already 1 if on sale before the data starts)."""
    if start <= 0:
        return np.ones(len(t))
    return np.clip((t - start) / years, 0, 1)


def market_by_fuel(rng):
    t = t_years(MONTHS)
    festive = 1 + 0.18 * np.isin(MONTHS.month, [10, 11]) - 0.06 * np.isin(MONTHS.month, [6, 7])
    covid = np.where((MONTHS >= "2020-04-01") & (MONTHS <= "2020-06-01"), 0.35, 1.0)
    total = 290_000 * (1 + 0.05 * t) * festive * covid * rng.normal(1, 0.03, len(t))

    # early stage of an S-curve: share = L / (1 + exp(-k (t - t0)))
    ev = 0.60 / (1 + np.exp(-0.55 * (t - 12.5)))
    hybrid = 0.010 + 0.002 * t
    cng = 0.06 + 0.012 * t
    diesel = np.clip(0.30 - 0.025 * t, 0.12, None)
    petrol = 1 - ev - hybrid - cng - diesel
    shares = pd.DataFrame({"EV": ev, "Petrol": petrol, "Diesel": diesel, "CNG": cng,
                           "Hybrid": hybrid}, index=MONTHS)
    shares = shares * rng.lognormal(0, 0.06, shares.shape)      # month-to-month noise per fuel
    shares = shares.div(shares.sum(axis=1), axis=0)
    return shares.mul(total, axis=0)


def company_share_of_fuel(rng, t):
    # Company X dominated EV early; share erodes once Competitor A and C ramp up
    ev = np.where(t < 2.5, 0.88, 0.88 - 0.0375 * (t - 2.5)) + rng.normal(0, 0.01, len(t))
    return {"EV": np.clip(ev, 0.3, 0.95), "Petrol": np.full(len(t), 0.11),
            "Diesel": np.full(len(t), 0.10), "CNG": np.full(len(t), 0.06),
            "Hybrid": np.zeros(len(t))}


def build_market(rng):
    units = market_by_fuel(rng)
    t = t_years(MONTHS)
    cx = company_share_of_fuel(rng, t)
    frames = []
    for fuel in FUELS:
        company = units[fuel].values * cx[fuel]
        rest = units[fuel].values - company
        if fuel == "EV":
            w = np.vstack([s * ramp(t, t_years(pd.DatetimeIndex([launch]))[0])
                           for launch, s in EV_COMPETITORS.values()])
            names = list(EV_COMPETITORS)
        else:
            w = np.repeat(rng.dirichlet(np.ones(len(ICE_COMPETITORS)) * 4)[:, None], len(t), axis=1)
            names = ICE_COMPETITORS
        w = w / w.sum(axis=0)
        frames.append(pd.DataFrame({"month": MONTHS, "brand": "Company X", "fuel_type": fuel, "units": company}))
        for name, wi in zip(names, w):
            frames.append(pd.DataFrame({"month": MONTHS, "brand": name, "fuel_type": fuel, "units": rest * wi}))
    df = pd.concat(frames, ignore_index=True)
    df["units"] = df["units"].round().astype(int)
    return df[df["units"] > 0].sort_values(["month", "brand", "fuel_type"]).reset_index(drop=True)


def ev_model_weights(t):
    city = np.where(t < 4.5, 1.0, np.clip(1.0 - 0.18 * (t - 4.5), 0.45, None))  # fading from mid-2022
    rng_ = 0.55 * ramp(t, 2.42)
    pro = 0.35 * ramp(t, 5.25, years=1.0)
    return {"X-City EV": city, "X-Range EV": rng_, "X-Pro EV": pro}


def build_company_models(market, rng):
    cx = (market[market["brand"] == "Company X"]
          .pivot(index="month", columns="fuel_type", values="units").fillna(0))
    t = t_years(cx.index)
    ev_w = ev_model_weights(t)
    parts = []
    for model, segment, fuel, launch, price, w in MODELS:
        weight = ev_w[model] if fuel == "EV" else np.full(len(t), w)
        weight = np.where(cx.index >= pd.Timestamp(launch), weight, 0.0)
        parts.append(pd.DataFrame({"month": cx.index, "model": model, "segment": segment,
                                   "fuel_type": fuel, "weight": weight,
                                   "price": price * (1 + 0.03 * t)}))
    df = pd.concat(parts, ignore_index=True)
    df["weight"] = df["weight"] / df.groupby(["month", "fuel_type"])["weight"].transform("sum")
    fuel_units = cx.stack().rename("fuel_units").reset_index()
    df = df.merge(fuel_units, on=["month", "fuel_type"])
    df["units"] = (df["weight"] * df["fuel_units"] * rng.normal(1, 0.03, len(df))).round()
    df["avg_price_usd"] = (df["price"] * rng.normal(1, 0.01, len(df))).round(-1)
    df = df[df["units"] > 0].astype({"units": int})
    return df[["month", "model", "segment", "fuel_type", "units", "avg_price_usd"]]


def build_reviews():
    # customer scores out of 5 per aspect. X-City EV is weak on range and charging speed.
    s = {
        "X-City EV":      {"range": 2.7, "charging": 2.9, "price": 4.2, "build_quality": 3.7, "features": 3.5},
        "X-Range EV":     {"range": 3.9, "charging": 3.6, "price": 3.8, "build_quality": 3.9, "features": 4.0},
        "X-Pro EV":       {"range": 4.4, "charging": 4.2, "price": 3.1, "build_quality": 4.3, "features": 4.5},
        "X-City Petrol":  {"price": 4.3, "build_quality": 3.8, "features": 3.2, "mileage": 3.9},
        "X-Trail Petrol": {"price": 3.9, "build_quality": 4.0, "features": 3.6, "mileage": 3.4},
        "X-Trail Diesel": {"price": 3.6, "build_quality": 4.0, "features": 3.6, "mileage": 4.2},
        "X-City CNG":     {"price": 4.4, "build_quality": 3.6, "features": 2.9, "mileage": 4.6},
    }
    n = {"X-City EV": 3120, "X-Range EV": 1480, "X-Pro EV": 410, "X-City Petrol": 5200,
         "X-Trail Petrol": 2900, "X-Trail Diesel": 2100, "X-City CNG": 1700}
    rows = [(m, a, v, n[m]) for m, d in s.items() for a, v in d.items()]
    return pd.DataFrame(rows, columns=["model", "aspect", "score", "n_reviews"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    market = build_market(rng)
    tables = {"market_sales": market,
              "company_models": build_company_models(market, rng),
              "model_reviews": build_reviews()}
    for name, df in tables.items():
        if "month" in df:
            df = df.assign(month=df["month"].dt.strftime("%Y-%m"))
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"[simulated] {name}.csv: {len(df):,} rows")


if __name__ == "__main__":
    main()
