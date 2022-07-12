import numpy as np
import datetime as dt

from portfolio_optimization.meta import *
from portfolio_optimization.utils.metrics import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.paths import *
from portfolio_optimization.bloomberg import *


def test_multi_period_portfolio():
    prices = load_prices(file=TEST_PRICES_PATH)

    N = 10
    periods = [(dt.date(2018, 1, 1), dt.date(2018, 3, 1)),
               (dt.date(2018, 3, 15), dt.date(2018, 5, 1)),
               (dt.date(2018, 5, 1), dt.date(2018, 8, 1))]

    mpp = MultiPeriodPortfolio()
    returns = np.array([])
    for i, period in enumerate(periods):
        assets = Assets(prices=prices, start_date=period[0], end_date=period[1], verbose=False)
        weights = rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - N)
        portfolio = Portfolio(weights=weights, assets=assets, pid=str(i))
        mpp.add(portfolio)
        returns = np.concatenate([returns, portfolio_returns(assets.returns, weights)])

    assert np.all((returns - mpp.returns) < 1e-10)
    assert abs(returns.mean() - mpp.mean) < 1e-10
    assert abs(returns.std(ddof=1) - mpp.std) < 1e-10
    assert abs(np.sqrt(np.sum(np.minimum(0, returns - returns.mean()) ** 2) / (len(returns) - 1))
               - mpp.downside_std) < 1e-10
    assert abs(mpp.annualized_mean / mpp.annualized_std
               - mpp.sharpe_ratio) < 1e-10
    assert abs(mpp.annualized_mean / mpp.annualized_downside_std
               - mpp.sortino_ratio) < 1e-10
    assert max_drawdown_slow(mpp.prices_compounded) == mpp.max_drawdown
    assert np.array_equal(mpp.fitness,
                          np.array([mpp.mean, -mpp.std]))
    mpp.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD)
    assert np.array_equal(mpp.fitness,
                          np.array([mpp.mean, -mpp.downside_std]))
    mpp.reset_fitness(fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN)
    assert np.array_equal(mpp.fitness,
                          np.array([mpp.mean,
                                    -mpp.downside_std,
                                    -mpp.max_drawdown]))
    assert len(mpp.assets_index) == len(periods)
    assert len(mpp.assets_names) == len(periods)
    assert mpp.composition.shape[1] == len(periods)
    mpp.reset_metrics()
    assert mpp._mean is None
    assert mpp._std is None
    mpp.plot_returns()
    mpp.plot_prices_compounded()
    mpp.plot_prices_uncompounded()
    mpp.plot_rolling_sharpe(days=20)

