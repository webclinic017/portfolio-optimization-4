import datetime as dt
import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.optimization.mean_variance import *
from portfolio_optimization.optimization.mean_semivariance import *
from portfolio_optimization.optimization.mean_cdar import *
from portfolio_optimization.utils.pre_seclection import *

np.random.seed(150)


def load_subset_assets():
    """
    Load Assets form multiple periods
    """
    assets = Assets(start_date=dt.date(2018, 1, 1), asset_missing_threshold=0.05)
    new_names = [assets.names[i] for i in np.random.choice(assets.asset_nb, 200, replace=False)]
    assets = Assets(start_date=dt.date(2018, 1, 1), end_date=dt.date(2019, 1, 1), names_to_keep=new_names)
    assets_pre_selected = pre_selection(assets=assets, k=100)
    assets_2018 = Assets(start_date=dt.date(2018, 1, 1), end_date=dt.date(2019, 1, 1),
                         names_to_keep=assets_pre_selected, name='2018')
    assets_2019 = Assets(start_date=dt.date(2019, 1, 1), end_date=dt.date(2020, 1, 1),
                         names_to_keep=assets_pre_selected, name='2019')
    assert np.array_equal(assets_2018.names, assets_2019.names)
    return assets_2018, assets_2019


def random_vs_mean_variance():
    """
    Compare the efficient frontier of the mean-variance optimization against portfolios of single asset and
    random portfolios
    """
    _, assets = load_subset_assets()

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
    assets_2018, assets_2019 = load_subset_assets()

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


def mean_variance_vs_mean_semivariance():
    """
    Compare the Efficient Frontier of the mean-variance against the mean-semivariance optimization
    """
    _, assets = load_subset_assets()

    population = Population()

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
                                 pid=f'mean_variance_{i}',
                                 name=str(i),
                                 tag='mean_variance'))

    # Efficient Frontier -- Mean Semivariance
    portfolios_weights = mean_semivariance(expected_returns=assets.mu,
                                           returns=assets.returns,
                                           returns_target=assets.mu,
                                           investment_type=InvestmentType.FULLY_INVESTED,
                                           weight_bounds=(0, None),
                                           population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 pid=f'mean_semivariance_{i}',
                                 name=str(i),
                                 tag='mean_semivariance'))

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SHARPE_RATIO)
    population.plot(x=Metrics.ANNUALIZED_DOWNSIDE_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SORTINO_RATIO)

    # Metrics
    max_sharpe = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe)
    print(max_sharpe.sharpe_ratio)

    max_sortino = population.max(metric=Metrics.SORTINO_RATIO)
    print(max_sortino)
    print(max_sortino.sortino_ratio)

    # Composition
    population.plot_composition(pids=[max_sharpe.pid, max_sortino.pid])


def mean_variance_vs_mean_cdar():
    """
    Compare the Efficient Frontier of the mean-variance against the mean-cdar optimization
    """
    _, assets = load_subset_assets()

    population = Population()

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
                                 pid=f'mean_variance_{i}',
                                 name=str(i),
                                 tag='mean_variance'))

    # Efficient Frontier -- Mean Cdar
    portfolios_weights = mean_cdar(expected_returns=assets.mu,
                                   returns=assets.returns,
                                   investment_type=InvestmentType.FULLY_INVESTED,
                                   weight_bounds=(0, None),
                                   beta=0.95,
                                   population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 pid=f'mean_cdar_{i}',
                                 name=str(i),
                                 tag='mean_cdar'))

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SHARPE_RATIO)
    population.plot(x=Metrics.ANNUALIZED_DOWNSIDE_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SORTINO_RATIO)
    population.plot(x=Metrics.MAX_DRAWDOWN,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.CALMAR_RATIO)
    population.plot(x=Metrics.CDAR_95,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.CDAR_95_RATIO)

    # Metrics
    max_sharpe = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe)
    print(max_sharpe.sharpe_ratio)

    max_cdar_ratio = population.max(metric=Metrics.CDAR_95_RATIO)
    print(max_cdar_ratio)
    print(max_cdar_ratio.sortino_ratio)

    # Composition
    population.plot_composition(pids=[max_sharpe.pid, max_cdar_ratio.pid])
