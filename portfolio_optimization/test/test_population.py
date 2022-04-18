import datetime as dt

from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *


def test_population():
    assets = Assets(date_from=dt.date(2019, 1, 1))

    # Create a population of portfolios with 3 objectives
    population = Population()
    for _ in range(100):
        weights = rand_weights(n=assets.asset_nb)
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN, assets=assets)
        population.append(portfolio)

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
    population.plot(x='annualized_downside_std', y='annualized_mu', z='max_drawdown', fronts=True)

    # Create a population of portfolios with 2 objectives
    population = Population()
    for _ in range(100):
        weights = rand_weights(n=assets.asset_nb)
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets)
        population.append(portfolio)

    population.plot(x='annualized_std', y='annualized_mu', fronts=True)

    # prices = load_bloomberg_prices(date_from=dt.date(2019, 1, 1))
    # prices=prices.iloc[:, :30]
    # assets = Assets(prices=prices)
