import pandas as pd
from pathlib import Path
from portfolio_optimization.utils.metrics import *
from portfolio_optimization.paths import TEST_FOLDER

returns = pd.read_csv(Path(TEST_FOLDER, 'data', 'returns.csv')).to_numpy().reshape((-1))


def is_close(a: float, b: float, precision: float = 1e-8):
    return abs(a - b) < precision


def test_semi_variance():
    assert is_close(semi_variance(returns), 2.0057127003099892e-05)


def test_kurtosis():
    assert is_close(kurtosis(returns), 1.993470262836003e-08)


def test_semi_kurtosis():
    assert is_close(semi_kurtosis(returns), 1.654455469800636e-08)


def test_mad():
    assert is_close(mad(returns), 0.003738102999822195)


def test_cvar():
    assert is_close(cvar(returns), 0.014433378234723564)


def test_value_at_risk():
    assert is_close(value_at_risk(returns), 0.00789712)


def test_worst_return():
    assert is_close(worst_return(returns), 0.049687264)


def test_first_lower_partial_moment():
    assert is_close(first_lower_partial_moment(returns), 0.0018690514999110974)
    assert is_close(first_lower_partial_moment(returns, min_acceptable_return=0), 0.0017230053656644036)


def test_entropic_risk_measure():
    assert is_close(entropic_risk_measure(returns), 2.995415784082025)
    assert is_close(entropic_risk_measure(returns, theta=0.5, beta=0.5), 0.34627370469589625)


def test_evar():
    e, theta = evar(returns)
    assert is_close(e, 0.02958216021229341)
    assert is_close(theta, 0.0071008766156466525)


def test_dar():
    assert is_close(dar(returns), 0.09762014000000052)
    assert is_close(dar(returns, compounded=True), 0.09816816006247986)


def test_cdar():
    assert is_close(cdar(returns), 0.1528201870611057)
    assert is_close(cdar(returns, compounded=True), 0.14635199737798482)


def test_max_drawdown():
    assert is_close(max_drawdown(returns), 0.28173365799999994)
    assert is_close(max_drawdown(returns, compounded=True), 0.24946912548114242)


def test_avg_drawdown():
    assert is_close(avg_drawdown(returns), 0.02907083392531529)
    assert is_close(avg_drawdown(returns, compounded=True), 0.029631700403305428)


def test_avg_drawdown():
    e, theta = edar(returns)
    assert is_close(e, 0.19849532531113231)
    assert is_close(theta, 0.03486095123355589)

import riskfolio as rp

def test_ulcer_index():
    assert is_close(ulcer_index(returns), 0.04889755862741813)
    assert is_close(ulcer_index(returns, compounded=True), 0.04852766892714257)


def test_gini_mean_difference():
    assert is_close(gini_mean_difference(returns), 0.005579067291230119)


import riskfolio as rp

rp.MAD()

