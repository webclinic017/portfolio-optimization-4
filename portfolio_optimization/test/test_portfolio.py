import numpy as np
import datetime as dt

from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *


def test_portfolio_metrics():
    assets = Assets(start_date=dt.date(2019, 1, 1))
    weights = rand_weights(n=assets.asset_nb)
    portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets)

    returns = portfolio_returns(assets.returns, weights)

    assert np.all((returns - portfolio.returns) < 1e-10)
    assert abs(returns.mean() - portfolio.mean) < 1e-10
    assert abs(returns.std(ddof=1) - portfolio.std) < 1e-10
    assert abs(np.sqrt(np.mean(np.minimum(0, returns) ** 2)) - portfolio.downside_std) < 1e-10
    assert abs(portfolio.mean / portfolio.std - portfolio.sharpe_ratio) < 1e-10
    assert abs(portfolio.mean / portfolio.downside_std - portfolio.sortino_ratio) < 1e-10
    assert np.array_equal(prices_rebased(portfolio.returns), portfolio.prices)
    assert max_drawdown_slow(portfolio.prices) == portfolio.max_drawdown
    assert np.array_equal(portfolio.fitness, np.array([portfolio.mean, -portfolio.std]))
    portfolio.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD)
    assert np.array_equal(portfolio.fitness, np.array([portfolio.mean, -portfolio.downside_std]))
    portfolio.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN)
    assert np.array_equal(portfolio.fitness, np.array([portfolio.mean, -portfolio.downside_std, -portfolio.max_drawdown]))
    portfolio.reset_metrics()
    assert portfolio._mean is None
    assert portfolio._std is None


def test_portfolio_dominate():
    assets = Assets(start_date=dt.date(2019, 1, 1))

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

