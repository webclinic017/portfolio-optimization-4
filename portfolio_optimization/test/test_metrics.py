import pandas as pd
from pathlib import Path
from portfolio_optimization.utils.metrics import *
from portfolio_optimization.paths import TEST_FOLDER

returns = pd.read_csv(Path(TEST_FOLDER, 'data', 'returns.csv')).to_numpy().reshape((-1))


def is_close(a: float, b: float, precision: float = 1e-8):
    return abs(a - b) < precision


def test_semi_variance():
    is_close(semi_variance(returns), 2.0057127003099892e-05)


def test_kurtosis():
    is_close(kurtosis(returns), 1.993470262836003e-08)


def test_semi_kurtosis():
    is_close(semi_kurtosis(returns), 1.654455469800636e-08)


def test_mad():
    is_close(mad(returns), 0.003738102999822195)


def test_cvar():
    is_close(cvar(returns), 0.014433378234723564)


def test_cdar():
    is_close(cdar(returns), 0.1528201870611057)
    is_close(cdar(returns, compounded=True), 0.14635199737798482)


def test_max_drawdown():
    is_close(max_drawdown(returns), 0.28173365799999994)
    is_close(max_drawdown(returns, compounded=True), 0.24946912548114242)


def test_value_at_risk():
    is_close(value_at_risk(returns), 0.00789712)


def test_worst_return():
    is_close(worst_return(returns), 0.049687264)


def test_first_lower_partial_moment():
    is_close(first_lower_partial_moment(returns), 0.0018690514999110974)
    is_close(first_lower_partial_moment(returns, min_acceptable_return=0), 0.0017230053656644036)


def test_entropic_risk_measure():
    is_close(entropic_risk_measure(returns), 2.995415784082025)
    is_close(entropic_risk_measure(returns, theta=0.5, beta=0.5), 0.34627370469589625)


def test_evar():
    e, theta = evar(returns)
    is_close(e, 0.02958216021229341)
    is_close(theta, 0.0071008766156466525)
