import datetime as dt

from portfolio_optimization.bloomberg.loader import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *


def test_population():
    prices = load_bloomberg_prices(date_from=dt.date(2019, 1, 1))
    assets = Assets(prices=prices)

    population = Population()

    for _ in range(10):
        weights = rand_weights(n=assets.asset_nb)
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN, assets=assets)
        population.append(portfolio)

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

    population.plot(x='mu', y='downside_std', z='max_drawdown', fronts=True)
    population.plot(x='mu', y='std', fronts=True)
