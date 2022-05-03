import datetime as dt
import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.optimization.mean_variance import *
from portfolio_optimization.utils.assets import *


def random_vs_mean_variance():
    """
    Compare the efficient frontier of the mean-variance optimization against portfolios of single asset and
    random portfolios
    """
    assets = load_assets_with_preselection(start_date=dt.date(2018, 1, 1),
                                           end_date=dt.date(2019, 1, 1),
                                           random_selection=200,
                                           pre_selection_number=100)

    population = Population()
    # Portfolios of one asset
    for i in range(assets.asset_nb):
        weights = np.zeros(assets.asset_nb)
        weights[i] = 1
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 tag='single_asset'))

    # Random portfolios
    for _ in range(10):
        weights = rand_weights_dirichlet(n=assets.asset_nb)
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 tag='random'))

    # Efficient Frontier -- Mean Variance
    portfolios_weights = mean_variance(expected_returns=assets.mu,
                                       cov=assets.cov,
                                       investment_type=InvestmentType.FULLY_INVESTED,
                                       weight_bounds=(0, None),
                                       population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 name=str(i),
                                 tag='mean_variance'))

    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN, color_scale=Metrics.SHARPE_RATIO)


def mean_variance_different_periods():
    """
    Compare the efficient frontier of the mean-variance optimization fitted on the train period (2018-2019) against the
    frontier tested on the test period (2019-2020)
    """
    assets_2018, assets_2019 = load_train_test_assets(train_period=(dt.date(2018, 1, 1), dt.date(2019, 1, 1)),
                                                      test_period=(dt.date(2019, 1, 1), dt.date(2020, 1, 1)),
                                                      random_selection=400,
                                                      pre_selection_number=100)

    population = Population()

    # Efficient Frontier -- Mean Variance -- Per period
    for assets in [assets_2018, assets_2019]:
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
    for portfolio in population.get_portfolios(tags=f'train_{assets_2018.name}'):
        population.add(Portfolio(weights=portfolio.weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets_2019,
                                 pid=f'test_{assets.name}_{portfolio.name}',
                                 name=portfolio.name,
                                 tag=f'test_{assets_2019.name}'))

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SHARPE_RATIO)
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale='name',
                    tags=[f'train_{assets_2018.name}', f'test_{assets_2019.name}'])
    population.plot_composition(tags=[f'train_{assets_2018.name}', f'train_{assets_2019.name}'])

    # Metrics
    max_sortino = population.max(metric=Metrics.SORTINO_RATIO)
    print(max_sortino.sortino_ratio)
