import numpy as np
import datetime as dt
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.utils.metrics import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.paths import *


def test_mpp_metrics():
    prices = load_prices(file=TEST_PRICES_PATH)
    n = 10
    periods = [(dt.date(2017, 1, 1), dt.date(2017, 3, 1)),
               (dt.date(2017, 3, 15), dt.date(2017, 5, 1)),
               (dt.date(2017, 5, 1), dt.date(2017, 8, 1))]
    returns = np.array([])
    portfolios = []
    for i, period in enumerate(periods):
        assets = Assets(prices=prices,
                        start_date=period[0],
                        end_date=period[1],
                        verbose=False)
        weights = rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - n)
        returns = np.concatenate([returns, portfolio_returns(assets.returns, weights)])
        portfolios.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'portfolio_{i}'))

    mpp = MultiPeriodPortfolio(portfolios=portfolios, name='mpp', tag='my_tag')
    assert len(mpp) == 3
    assert np.all((returns - mpp.returns) < 1e-10)
    assert abs(returns.mean() - mpp.mean) < 1e-10
    assert abs(returns.std(ddof=1) - mpp.std) < 1e-10
    assert abs(np.sqrt(np.sum(np.minimum(0, returns - returns.mean()) ** 2) / (len(returns) - 1))
               - mpp.semistd) < 1e-10
    assert abs(mpp.annualized_mean / mpp.annualized_std
               - mpp.sharpe_ratio) < 1e-10
    assert abs(mpp.annualized_mean / mpp.annualized_semistd
               - mpp.sortino_ratio) < 1e-10
    assert max_drawdown_slow(mpp.cumulative_returns) == mpp.max_drawdown
    assert np.array_equal(mpp.fitness,
                          np.array([mpp.mean, -mpp.std]))
    mpp.fitness_metrics= [Metrics.MEAN, Metrics.DOWNSIDE_STD]
    assert np.array_equal(mpp.fitness,
                          np.array([mpp.mean, -mpp.semistd]))
    mpp.fitness_metrics= [Metrics.MEAN, Metrics.DOWNSIDE_STD, Metrics.MAX_DRAWDOWN]
    assert np.array_equal(mpp.fitness,
                          np.array([mpp.mean,
                                    -mpp.semistd,
                                    -mpp.max_drawdown]))
    assert len(mpp.assets_index) == len(periods)
    assert len(mpp.assets_names) == len(periods)
    assert mpp.composition.shape[1] == len(periods)
    mpp.reset()
    assert mpp.__dict__.get('mean') is None
    assert mpp.__dict__.get('str') is None
    assert mpp.plot_returns(show=False)
    assert mpp.plot_cumulative_returns(show=False)
    assert mpp.plot_cumulative_returns_uncompounded(show=False)
    assert mpp.plot_rolling_sharpe(days=20, show=False)
    assert isinstance(mpp.composition, pd.DataFrame)
    assert mpp.plot_composition(show=False)
    assert isinstance(mpp.summary(), pd.Series)


def test_mpp_magic_methods():
    prices = load_prices(file=TEST_PRICES_PATH)
    n = 10
    periods = [(dt.date(2017, 1, 1), dt.date(2017, 3, 1)),
               (dt.date(2017, 3, 15), dt.date(2017, 5, 1)),
               (dt.date(2017, 5, 1), dt.date(2017, 8, 1))]

    mpp = MultiPeriodPortfolio()
    for i, period in enumerate(periods):
        assets = Assets(prices=prices,
                        start_date=period[0],
                        end_date=period[1],
                        verbose=False)
        portfolio = Portfolio(weights=rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - n),
                              assets=assets,
                              name=str(i))
        mpp.append(portfolio)

    assert len(mpp) == 3
    assert mpp[1] == mpp.portfolios[1]
    for i, p in enumerate(mpp):
        assert p.name == str(i)
    p_1 = mpp[1]
    assert mpp == mpp
    assert p_1 in mpp
    assert 3 not in mpp
    assert -mpp[1] == -p_1
    assert abs(mpp)[1] == abs(p_1)
    assert round(mpp, 2)[1] == round(p_1, 2)
    assert np.floor(mpp)[1] == np.floor(p_1)
    assert np.trunc(mpp)[1] == np.trunc(p_1)
    assert (mpp + mpp)[1] == p_1 * 2
    assert (mpp - mpp * 0.5)[1] == p_1 * 0.5
    assert (mpp - mpp * 0.4)[1] != p_1 * 0.5
    assert (mpp - mpp * 0.4)[1] != p_1 * 0.5
    assert (mpp / 2)[1] == p_1 * 0.5
    assert (mpp // 2)[1] == p_1 // 2
    del mpp[1]
    assert p_1 not in mpp
    assert len(mpp) == 2
    mpp[1] = p_1
    assert p_1 in mpp
    try:
        mpp[0] = p_1
        raise
    except ValueError:
        pass
    mpp.portfolios = [mpp[0], p_1]
    assert len(mpp) == 2
    assert mpp[0] != p_1
    assert mpp[1] == p_1
    try:
        mpp.portfolios = [p_1, p_1, p_1]
        raise
    except ValueError:
        pass
    assert len(mpp) == 2
    assert mpp[0] != p_1
    assert mpp[1] == p_1
