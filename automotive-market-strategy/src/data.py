"""Load and validate the three input tables."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
COMPANY = "Company X"


def _read(name, required):
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/generate_case_data.py first")
    df = pd.read_csv(path)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name}.csv is missing columns {missing}")
    if "month" in df:
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
        df["year"] = df["month"].dt.year
    return df


def load_all():
    market = _read("market_sales", ["month", "brand", "fuel_type", "units"])
    models = _read("company_models", ["month", "model", "segment", "fuel_type", "units", "avg_price_usd"])
    reviews = _read("model_reviews", ["model", "aspect", "score", "n_reviews"])
    validate(market, models, reviews)
    return market, models, reviews


def validate(market, models, reviews):
    problems = []
    if (market["units"] < 0).any() or (models["units"] < 0).any():
        problems.append("negative unit sales")
    if market.duplicated(["month", "brand", "fuel_type"]).any():
        problems.append("duplicate month/brand/fuel rows in market_sales")
    if not reviews["score"].between(1, 5).all():
        problems.append("review scores outside 1-5")
    if models.groupby("model")["fuel_type"].nunique().max() > 1:
        problems.append("a model is listed with more than one fuel type")

    # company model totals should add up to the company's market totals (allow 5% rounding/noise)
    cx = market[market["brand"] == COMPANY].groupby(["year", "fuel_type"])["units"].sum()
    md = models.groupby(["year", "fuel_type"])["units"].sum()
    gap = ((md - cx).abs() / cx).dropna()
    if (gap > 0.05).any():
        problems.append(f"model sales don't match company totals (max gap {gap.max():.1%})")

    if problems:
        raise ValueError("data validation failed: " + "; ".join(problems))
