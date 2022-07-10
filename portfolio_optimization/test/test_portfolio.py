import numpy as np
import datetime as dt

from portfolio_optimization.meta import *
from portfolio_optimization.utils.metrics import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.paths import *
from portfolio_optimization.bloomberg.loader import *


def test_portfolio_metrics():
    prices = load_prices(file=TEST_PRICES_PATH)

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices, start_date=start_date)
    N = 10
    weights = rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - N)
    portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets)

    returns = portfolio_returns(assets.returns, weights)

    assert np.all((returns - portfolio.returns) < 1e-10)
    assert abs(returns.mean() - portfolio.mean) < 1e-10
    assert abs(returns.std(ddof=1) - portfolio.std) < 1e-10
    assert abs(np.sqrt(np.sum(np.minimum(0, returns - returns.mean()) ** 2) / (len(returns) - 1))
               - portfolio.downside_std) < 1e-10
    assert abs(portfolio.annualized_mean / portfolio.annualized_std - portfolio.sharpe_ratio) < 1e-10
    assert abs(portfolio.annualized_mean / portfolio.annualized_downside_std - portfolio.sortino_ratio) < 1e-10
    assert max_drawdown_slow(portfolio.prices_compounded) == portfolio.max_drawdown
    assert np.array_equal(portfolio.fitness, np.array([portfolio.mean, -portfolio.std]))
    portfolio.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD)
    assert np.array_equal(portfolio.fitness, np.array([portfolio.mean, -portfolio.downside_std]))
    portfolio.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN)
    assert np.array_equal(portfolio.fitness,
                          np.array([portfolio.mean, -portfolio.downside_std, -portfolio.max_drawdown]))
    assert len(portfolio.assets_index) == N
    assert len(portfolio.assets_names) == N
    assert len(portfolio.composition) == N
    idx = np.nonzero(weights)[0]
    assert np.array_equal(portfolio.assets_index, idx)
    names_1 = np.array(assets.prices.columns[idx])
    assert np.array_equal(portfolio.assets_names, names_1)
    names_2 = portfolio.composition.index.to_numpy()
    names_2.sort()
    names_1.sort()
    assert np.array_equal(names_1, names_2)
    portfolio.reset_metrics()
    assert portfolio._mean is None
    assert portfolio._std is None
    portfolio.plot_rolling_sharpe(days=20)


def test_portfolio_dominate():
    prices = load_prices(file=TEST_PRICES_PATH)

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,  start_date=start_date)

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
