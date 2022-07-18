import datetime as dt

from portfolio_optimization.meta import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.paths import *
from portfolio_optimization.bloomberg import *


def test_population():
    prices = load_prices(file=TEST_PRICES_PATH)

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    verbose=False)
    # Create a population of portfolios with 3 objectives
    population = Population()
    for i in range(100):
        weights = rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - 10)
        portfolio = Portfolio(weights=weights,
                              fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN,
                              assets=assets,
                              name=f'portfolio_{i}')
        population.add(portfolio)

    # test non-dominated sorting into fronts
    assert sorted([i for j in population.fronts for i in j]) == list(range(population.length))
    for i, front in enumerate(population.fronts):
        dominates = False
        if i == len(population.fronts) - 1:
            dominates = True
        for idx_1 in front:
            for j in range(i + 1, len(population.fronts)):
                for idx_2 in population.fronts[j]:
                    assert not population.portfolios[idx_2].dominates(population.portfolios[idx_1])
                    if population.portfolios[idx_1].dominates(population.portfolios[idx_2]):
                        dominates = True
        assert dominates

    # test plots
    population.plot_metrics(x=Metrics.ANNUALIZED_DOWNSIDE_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            z=Metrics.MAX_DRAWDOWN,
                            fronts=True)

    # Create a population of portfolios with 2 objectives
    population = Population()
    for i in range(10):
        weights = rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - 10)
        portfolio = Portfolio(weights=weights,
                              fitness_type=FitnessType.MEAN_STD,
                              assets=assets,
                              name=f'portfolio_{i}',
                              tag='random')
        population.add(portfolio)

    # Add the multi period portfolio
    periods = [(dt.date(2018, 1, 1), dt.date(2018, 3, 1)),
               (dt.date(2018, 3, 15), dt.date(2018, 5, 1)),
               (dt.date(2018, 5, 1), dt.date(2018, 8, 1))]

    mpp = MultiPeriodPortfolio(name='mmp')
    for i, period in enumerate(periods):
        assets = Assets(prices=prices,
                        start_date=period[0],
                        end_date=period[1],
                        verbose=False)
        weights = rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - 5)
        portfolio = Portfolio(weights=weights,
                              assets=assets,
                              name=f'portfolio_period_{i}',
                              tag='mpp')
        population.add(portfolio)
        mpp.add(portfolio)
    population.add(mpp)

    assert population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                                   y=Metrics.ANNUALIZED_MEAN,
                                   fronts=True,
                                   show=False)

    assert population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                                   y=Metrics.ANNUALIZED_MEAN,
                                   hover_metrics=[Metrics.SHARPE_RATIO],
                                   tags='random',
                                   title='Portfolios -- with sharpe ration',
                                   show=False)

    assert (population.min(metric=Metrics.ANNUALIZED_MEAN).annualized_mean
            <= population.max(metric=Metrics.ANNUALIZED_MEAN).annualized_mean)

    # get
    assert population.get(name='portfolio_2') == population.iloc(2)

    # composition
    assert population.composition()
    assert population.plot_composition(show=False)
