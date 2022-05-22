import datetime as dt

from portfolio_optimization.meta import *
from portfolio_optimization.meta import Metrics, FitnessType
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization.mean_variance import *
from portfolio_optimization.optimization.mean_semivariance import *
from portfolio_optimization.utils.assets import *


def mean_variance_vs_mean_semivariance():
    """
    Compare the Efficient Frontier of the mean-variance against the mean-semivariance optimization
    """
    assets = load_assets(start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=200,
                         pre_selection_number=100)

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
    print(max_sharpe.sharpe_ratio)

    max_sortino = population.max(metric=Metrics.SORTINO_RATIO)
    print(max_sortino.sortino_ratio)

    # Composition
    population.plot_composition(pids=[max_sharpe.pid, max_sortino.pid])
