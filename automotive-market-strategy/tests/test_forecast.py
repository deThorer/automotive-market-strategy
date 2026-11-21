import numpy as np
import pandas as pd
import pytest

from src import forecast as fc


def test_logistic_fit_recovers_known_parameters():
    t = np.linspace(2018, 2025, 84)
    true = fc.Logistic(L=0.5, k=0.6, t0=2031)
    fit = fc.fit_logistic(t, true.predict(t), L=0.5)
    assert fit.k == pytest.approx(0.6, rel=1e-6)
    assert fit.t0 == pytest.approx(2031, rel=1e-6)


def test_logistic_rejects_share_above_saturation():
    with pytest.raises(ValueError):
        fc.fit_logistic(np.array([2020, 2021]), np.array([0.2, 0.4]), L=0.3)


def test_mape():
    assert fc.mape([100, 200], [110, 180]) == pytest.approx(10.0)


def test_total_forecast_skips_2020():
    # flat market except a crash in 2020 -> growth should be ~0, not pulled down
    rows = [{"year": y, "units": 100 if y != 2020 else 30} for y in range(2018, 2025)]
    market = pd.DataFrame(rows)
    total, growth = fc.total_market_forecast(market, [2025])
    assert growth == pytest.approx(0, abs=1e-9)
    assert total[2025] == pytest.approx(100)
