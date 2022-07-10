import datetime as dt

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization.mean_variance import *
from portfolio_optimization.utils.assets import *
from portfolio_optimization.bloomberg.loader import *

if __name__ == '__main__':
    """
    Compare the efficient frontier of the mean-variance optimization fitted on the train period (2018-2019) against the
    frontier tested on the test period (2019-2020)
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    assets_train, assets_test = load_train_test_assets(prices=prices,
                                                       train_period=(dt.date(2018, 1, 1), dt.date(2019, 1, 1)),
                                                       test_period=(dt.date(2019, 1, 1), dt.date(2020, 1, 1)),
                                                       random_selection=400,
                                                       pre_selection_number=100)

    population = Population()

    # Efficient Frontier -- Mean Variance -- Per period
    for assets in [assets_train, assets_test]:
        portfolios_weights = mean_variance(expected_returns=assets.mu,
                                           cov=assets.cov,
                                           investment_type=InvestmentType.FULLY_INVESTED,
                                           weight_bounds=(0, None),
                                           population_size=30)
        for i, weights in enumerate(portfolios_weights):
            population.add(Portfolio(weights=weights,
                                     fitness_type=FitnessType.MEAN_STD,
                                     assets=assets,
                                     pid=f'train_{assets.name}_{i}',
                                     name=str(i),
                                     tag=f'train_{assets.name}'))

    # Test the portfolios on the test period
    for portfolio in population.get_portfolios(tags=f'train_{assets_train.name}'):
        population.add(Portfolio(weights=portfolio.weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets_test,
                                 pid=f'test_{assets.name}_{portfolio.name}',
                                 name=portfolio.name,
                                 tag=f'test_{assets_test.name}'))

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SHARPE_RATIO)
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale='name',
                    tags=[f'train_{assets_train.name}', f'test_{assets_test.name}'])
    population.plot_composition(tags=[f'train_{assets_train.name}', f'train_{assets_test.name}'])

    # Metrics
    max_sortino = population.max(metric=Metrics.SORTINO_RATIO)
    print(max_sortino.sortino_ratio)
