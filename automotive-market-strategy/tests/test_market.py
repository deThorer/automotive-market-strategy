import pandas as pd
import pytest

from src import market as mk


@pytest.fixture
def market():
    rows = []
    for year, cx_ev, rival_ev, petrol in [(2023, 80, 20, 900), (2024, 70, 30, 900)]:
        rows += [{"month": pd.Timestamp(f"{year}-01-01"), "year": year, "brand": "Company X", "fuel_type": "EV", "units": cx_ev},
                 {"month": pd.Timestamp(f"{year}-01-01"), "year": year, "brand": "Rival", "fuel_type": "EV", "units": rival_ev},
                 {"month": pd.Timestamp(f"{year}-01-01"), "year": year, "brand": "Company X", "fuel_type": "Petrol", "units": petrol}]
    return pd.DataFrame(rows)


def test_ev_segment_share(market):
    seg = mk.ev_segment_shares(market)
    assert seg.loc[2024, "Company X"] == 70.0
    assert seg.loc[2023, "Rival"] == 20.0


def test_company_overview(market):
    ov = mk.company_overview(market)
    assert ov.loc[2024, "company_share_of_ev_segment_pct"] == 70.0
    assert ov.loc[2024, "ev_pct_of_company_sales"] == pytest.approx(100 * 70 / 970, abs=0.01)


def test_growth(market):
    units, _ = mk.yearly_fuel_mix(market)
    g = mk.growth_by_fuel(units, 2023, 2024)
    assert g.loc["EV", "cagr_pct"] == 0.0     # 100 -> 100
    assert g.loc["Petrol", "last_yoy_pct"] == 0.0
