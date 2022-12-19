import numpy as np
import datetime as dt
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.utils.sorting import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.paths import *


def test_portfolio_metrics():
    prices = load_prices(file=TEST_PRICES_PATH)

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    verbose=False)
    n = 10
    weights = rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - n)
    portfolio = Portfolio(weights=weights,
                          assets=assets,
                          name='portfolio_1')
    returns = portfolio_returns(assets.returns, weights)
    assert len(portfolio) == 10
    assert_is_close(returns, portfolio.returns)
    assert_is_close(returns.mean(), portfolio.mean)
    assert_is_close(returns.std(ddof=1), portfolio.std)
    assert_is_close(np.sqrt(np.sum(np.minimum(0, returns - returns.mean()) ** 2) / (len(returns) - 1)),
                    portfolio.semi_std)
    assert_is_close(portfolio.mean / portfolio.std, portfolio.sharpe_ratio)
    assert_is_close(portfolio.mean / portfolio.semi_std, portfolio.sortino_ratio)
    assert_is_close(portfolio.fitness, np.array([portfolio.mean, -portfolio.variance]))
    portfolio.fitness_metrics = [Perf.MEAN, RiskMeasure.SEMI_STD]
    assert_is_close(portfolio.fitness, np.array([portfolio.mean, -portfolio.semi_std]))
    portfolio.fitness_metrics = [Perf.MEAN, RiskMeasure.SEMI_STD, RiskMeasure.MAX_DRAWDOWN]
    assert_is_close(portfolio.fitness, np.array([portfolio.mean, -portfolio.semi_std, -portfolio.max_drawdown]))
    assert len(portfolio.assets_index) == n
    assert len(portfolio.assets_names) == n
    assert len(portfolio.composition) == n
    idx = np.nonzero(weights)[0]
    assert_is_close(portfolio.assets_index, idx)
    names_1 = np.array(assets.prices.columns[idx])
    assert np.array_equal(portfolio.assets_names, names_1)
    names_2 = portfolio.composition.index.to_numpy()
    names_2.sort()
    names_1.sort()
    assert np.array_equal(names_1, names_2)
    portfolio.summary()
    portfolio.reset()
    assert portfolio.plot_returns(show=False)
    assert portfolio.plot_cumulative_returns(show=False)
    assert portfolio.plot_rolling_sharpe(days=20, show=False)
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition(show=False)
    assert isinstance(portfolio.summary(), pd.Series)
    assert isinstance(portfolio.summary(formatted=False), pd.Series)
    assert portfolio.get_weight(asset_name=portfolio.assets_names[5])


def test_portfolio_magic_methods():
    prices = load_prices(file=TEST_PRICES_PATH)

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    verbose=False)
    ptf_1 = Portfolio(weights=rand_weights(n=assets.asset_nb),
                      assets=assets)
    ptf_2 = Portfolio(weights=rand_weights(n=assets.asset_nb),
                      assets=assets)

    assert len(ptf_1) == assets.asset_nb
    ptf = ptf_1 + ptf_2
    assert np.array_equal(ptf.weights, ptf_1.weights + ptf_2.weights)
    ptf = ptf_1 - ptf_2
    assert np.array_equal(ptf.weights, ptf_1.weights - ptf_2.weights)
    ptf = -ptf_1
    assert np.array_equal(ptf.weights, -ptf_1.weights)
    ptf = ptf_1 * 2.3
    assert np.array_equal(ptf.weights, 2.3 * ptf_1.weights)
    ptf = 2.3 * ptf_1
    assert np.array_equal(ptf.weights, 2.3 * ptf_1.weights)
    ptf = ptf_1 / 2.3
    assert np.array_equal(ptf.weights, ptf_1.weights / 2.3)
    ptf = abs(ptf_1)
    assert np.array_equal(ptf.weights, abs(ptf_1.weights))
    ptf = round(ptf_1, 2)
    assert np.array_equal(ptf.weights, np.round(ptf_1.weights, 2))
    ptf = np.floor(ptf_1)
    assert np.array_equal(ptf.weights, np.floor(ptf_1.weights))
    ptf = np.trunc(ptf_1)
    assert np.array_equal(ptf.weights, np.trunc(ptf_1.weights))
    ptf = ptf_1 // 2
    assert np.array_equal(ptf.weights, ptf_1.weights // 2)

    assert hash(ptf_1) == hash(ptf_1)
    assert hash(ptf_1) != hash(ptf_2)
    assert ptf_1 == ptf_1
    assert ptf_1 != ptf_2
    assert (ptf_1 > ptf_2) is ptf_1.dominates(ptf_2)
    assert (ptf_1 < ptf_2) is ptf_2.dominates(ptf_1)


def test_portfolio_dominate():
    prices = load_prices(file=TEST_PRICES_PATH)

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    verbose=False)

    for _ in range(1000):
        weights_1 = rand_weights(n=assets.asset_nb)
        weights_2 = rand_weights(n=assets.asset_nb)
        portfolio_1 = Portfolio(weights=weights_1,
                                fitness_metrics=[Perf.MEAN, RiskMeasure.SEMI_STD, RiskMeasure.MAX_DRAWDOWN],
                                assets=assets)
        portfolio_2 = Portfolio(weights=weights_2,
                                fitness_metrics=[Perf.MEAN, RiskMeasure.SEMI_STD, RiskMeasure.MAX_DRAWDOWN],
                                assets=assets)

        # Doesn't dominate itself (same front)
        assert portfolio_1.dominates(portfolio_1) is False
        assert dominate_slow(portfolio_1.fitness, portfolio_2.fitness) == portfolio_1.dominates(portfolio_2)


def test_portfolio_risk_contribution():
    prices = load_prices(file=TEST_PRICES_PATH)
    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    verbose=False)
    weights = rand_weights(n=assets.asset_nb)
    portfolio = Portfolio(weights=weights, assets=assets)
    rc = portfolio.risk_contribution(risk_measure=RiskMeasure.CVAR)
    res = np.array([0.0035595, 0.00312922, 0.00060825, 0.00074254, 0.00340565,
                    0.00156109, 0.00330621, 0.00133641, 0.00307977, 0.00197139,
                    0.00174254])
    assert_is_close(rc, res)


def test_portfolio_metrics():
    prices = load_prices(file=TEST_PRICES_PATH)
    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    verbose=False)
    weights = rand_weights(n=assets.asset_nb)
    portfolio = Portfolio(weights=weights, assets=assets)
    for enu in [Perf, RiskMeasure, Ratio]:
        for e in enu:
            assert getattr(portfolio, e.value)
