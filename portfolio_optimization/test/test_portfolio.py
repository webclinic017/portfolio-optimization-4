import numpy as np
import datetime as dt
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.utils.metrics import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.utils.sorting import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.paths import *
from portfolio_optimization.bloomberg.loader import *


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
    assert np.all((returns - portfolio.returns) < 1e-10)
    assert abs(returns.mean() - portfolio.mean) < 1e-10
    assert abs(returns.std(ddof=1) - portfolio.std) < 1e-10
    assert abs(np.sqrt(np.sum(np.minimum(0, returns - returns.mean()) ** 2) / (len(returns) - 1))
               - portfolio.downside_std) < 1e-10
    assert abs(portfolio.annualized_mean / portfolio.annualized_std - portfolio.sharpe_ratio) < 1e-10
    assert abs(portfolio.annualized_mean / portfolio.annualized_downside_std - portfolio.sortino_ratio) < 1e-10
    assert max_drawdown_slow(portfolio.cumulative_returns) == portfolio.max_drawdown
    assert np.array_equal(portfolio.fitness, np.array([portfolio.mean, -portfolio.std]))
    portfolio.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD)
    assert np.array_equal(portfolio.fitness, np.array([portfolio.mean, -portfolio.downside_std]))
    portfolio.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN)
    assert np.array_equal(portfolio.fitness,
                          np.array([portfolio.mean, -portfolio.downside_std, -portfolio.max_drawdown]))
    assert len(portfolio.assets_index) == n
    assert len(portfolio.assets_names) == n
    assert len(portfolio.composition) == n
    idx = np.nonzero(weights)[0]
    assert np.array_equal(portfolio.assets_index, idx)
    names_1 = np.array(assets.prices.columns[idx])
    assert np.array_equal(portfolio.assets_names, names_1)
    names_2 = portfolio.composition.index.to_numpy()
    names_2.sort()
    names_1.sort()
    assert np.array_equal(names_1, names_2)
    portfolio.summary()
    portfolio.reset()
    assert portfolio.__dict__.get('mean') is None
    assert portfolio.__dict__.get('std') is None
    assert portfolio.plot_returns(show=False)
    assert portfolio.plot_cumulative_returns(show=False)
    assert portfolio.plot_cumulative_returns_uncompounded(show=False)
    assert portfolio.plot_rolling_sharpe(days=20, show=False)
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition(show=False)
    assert isinstance(portfolio.summary(), pd.core.series.Series)
    assert isinstance(portfolio.summary(formatted=False), pd.core.series.Series)
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
    assert (ptf_1 < ptf_2) is not ptf_1.dominates(ptf_2)
    assert (ptf_1 > ptf_2) is (ptf_2 < ptf_1)
    assert (ptf_1 < ptf_2) is (ptf_2 > ptf_1)
    assert (ptf_1 >= ptf_2) is (ptf_1 > ptf_2)
    assert (ptf_1 <= ptf_2) is (ptf_1 < ptf_2)


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
                                fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN,
                                assets=assets)
        portfolio_2 = Portfolio(weights=weights_2,
                                fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN,
                                assets=assets)

        # Doesn't dominate itself (same front)
        assert portfolio_1.dominates(portfolio_1) is False
        assert dominate_slow(portfolio_1.fitness, portfolio_2.fitness) == portfolio_1.dominates(portfolio_2)
