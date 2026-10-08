"""No-look-ahead and orientation checks for alternative factors."""

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.alternative_factors import (
    alternative_factors,
    low_volatility_factor,
    reversal_factor,
)


def sample():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    t = np.arange(24)
    return pd.DataFrame(
        {"a": 0.01 + t * 0.0001, "b": (-1.0) ** t * 0.08,
         "c": 0.025 * np.sin(t), "d": 0.03 * np.cos(t),
         "e": 0.005 + t * 0.0002, "f": 0.04 * np.sin(2 * t)}, index=dates
    )


def test_factors_have_correct_names_and_warmup():
    result = alternative_factors(sample())
    assert list(result) == ["LOWVOL", "REV"]
    assert result["LOWVOL"].iloc[:12].isna().all()
    assert result["LOWVOL"].iloc[12:].notna().all()
    assert result["REV"].iloc[0] != result["REV"].iloc[0]


def test_no_contemporaneous_signal_leakage():
    data = sample()
    changed = data.copy()
    changed.iloc[-1] = 0.9
    base = alternative_factors(data)
    altered = alternative_factors(changed)
    pd.testing.assert_frame_equal(base.iloc[:-1], altered.iloc[:-1])


def test_invalid_window():
    with pytest.raises(ValueError):
        low_volatility_factor(sample(), window=2)
