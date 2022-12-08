import datetime as dt
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.paths import *
from copy import copy


def load_population() -> Population:
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
                              fitness_metrics=[Metrics.MEAN, Metrics.SEMISTD, Metrics.MAX_DRAWDOWN],
                              assets=assets,
                              name=str(i))
        population.append(portfolio)

    return population


def load_multi_period_portfolio() -> MultiPeriodPortfolio:
    prices = load_prices(file=TEST_PRICES_PATH)

    # Add the multi period portfolio
    periods = [(dt.date(2017, 1, 1), dt.date(2017, 3, 1)),
               (dt.date(2017, 3, 15), dt.date(2017, 5, 1)),
               (dt.date(2017, 5, 1), dt.date(2017, 8, 1))]

    mpp = MultiPeriodPortfolio(name='mmp',
                               fitness_metrics=[Metrics.MEAN, Metrics.SEMISTD, Metrics.MAX_DRAWDOWN],
                               )
    for i, period in enumerate(periods):
        assets = Assets(prices=prices,
                        start_date=period[0],
                        end_date=period[1],
                        verbose=False)
        portfolio = Portfolio(weights=rand_weights(n=assets.asset_nb, zeros=assets.asset_nb - 5),
                              assets=assets,
                              name=f'ptf_period_{i}')
        mpp.append(portfolio)
    return mpp


def test_magic_methods():
    population = load_population()
    assert population.portfolios == list(population)
    assert len(population) == 100
    assert population[0].name == '0'
    assert population[-1].name == '99'
    assert len(population[1:3]) == 2
    for i, ptf in enumerate(population):
        assert ptf.name == str(i)
    ptf = population[5]
    assert ptf in population
    assert population.get(name='5') == ptf
    del population[5]
    assert len(population) == 99
    assert ptf not in population
    population.append(ptf)
    assert len(population) == 100
    assert ptf in population
    try:
        population.append(ptf)
        raise
    except KeyError:
        pass
    assert len(population) == 100
    ptfs = list(population).copy()
    population.portfolios = ptfs
    assert list(population) == ptfs
    ptfs.append(ptf)
    try:
        population.portfolios = ptfs
        raise
    except KeyError:
        pass
    assert len(population) == 100
    ptf = population[10]
    population[10] = ptf
    assert len(population) == 100
    try:
        population[10] = population[11]
        raise
    except KeyError:
        pass
    assert len(population) == 100
    new_ptf = population[11]
    try:
        new_ptf.name = 'new_name'
        raise
    except AttributeError:
        pass
    new_ptf_copy = copy(new_ptf)
    new_ptf_copy.name = 'new_name'
    population[10] = new_ptf_copy
    assert len(population) == 100
    assert population[10] == new_ptf
    ptf = copy(new_ptf)
    ptf.name = 'different_fitness'
    ptf.fitness_metrics = [Metrics.MEAN, Metrics.SEMISTD, Metrics.SORTINO_RATIO]
    try:
        population.append(ptf)
        raise
    except ValueError:
        pass


def test_non_dominated_sorting():
    population = load_population()

    assert sorted([i for j in population.fronts for i in j]) == list(range(len(population)))
    for i, front in enumerate(population.fronts):
        dominates = False
        if i == len(population.fronts) - 1:
            dominates = True
        for idx_1 in front:
            for j in range(i + 1, len(population.fronts)):
                for idx_2 in population.fronts[j]:
                    assert not population[idx_2].dominates(population[idx_1])
                    if population[idx_1].dominates(population[idx_2]):
                        dominates = True
        assert dominates


def test_plot():
    population = load_population()

    assert population.plot_metrics(x=Metrics.ANNUALIZED_SEMISTD,
                                   y=Metrics.ANNUALIZED_MEAN,
                                   z=Metrics.MAX_DRAWDOWN,
                                   fronts=True,
                                   show=False)


def test_multi_period_portfolio():
    population = load_population()
    mpp = load_multi_period_portfolio()
    population.append(mpp)

    assert population.fronts
    assert len(population) == 101

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

    # composition
    assert isinstance(population.composition(), pd.DataFrame)
    assert population.plot_composition(show=False)
