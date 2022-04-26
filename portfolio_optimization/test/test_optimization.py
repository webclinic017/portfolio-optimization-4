import datetime as dt
import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization.mean_variance_optimization import *
from portfolio_optimization.optimization.mean_semivariance_optimization import *
from portfolio_optimization.utils.pre_seclection import *

np.random.seed(150)


def test():
    assets = Assets(start_date=dt.date(2018, 1, 1), asset_missing_threshold=0.05)
    new_names = [assets.names[i] for i in np.random.choice(assets.asset_nb, 200, replace=False)]
    assets = Assets(start_date=dt.date(2018, 1, 1), end_date=dt.date(2019, 1, 1), names_to_keep=new_names)
    assets_pre_selected = pre_selection(assets=assets, k=100)
    assets = Assets(start_date=dt.date(2018, 1, 1), end_date=dt.date(2019, 1, 1), names_to_keep=assets_pre_selected)
    assets_19_20 = Assets(start_date=dt.date(2019, 1, 1), end_date=dt.date(2020, 1, 1),
                          names_to_keep=assets_pre_selected)
    assert np.array_equal(assets.names, assets_19_20.names)

    population = Population()
    """
    # Portfolios of one asset
    for i in range(assets.asset_nb):
        weights = np.zeros(assets.asset_nb)
        weights[i] = 1
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets, tag='single asset')
        population.append(portfolio)

    # Random portfolios
    for _ in range(10):
        weights = rand_weights_dirichlet(n=assets.asset_nb)
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets, tag='random')
        population.append(portfolio)
    """
    # Pareto optimal portfolios
    optimal_weights = mean_variance_optimization(mu=assets.mu,
                                                 cov=assets.cov,
                                                 investment_type=InvestmentType.FULLY_INVESTED,
                                                 weight_bounds=(0, None),
                                                 population_size=100)
    for i, weights in enumerate(optimal_weights):
        portfolio = Portfolio(weights=weights,
                              fitness_type=FitnessType.MEAN_STD,
                              assets=assets,
                              name=str(i),
                              tag='markowitz')
        population.add(portfolio)

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN, color_scale='sharpe_ratio')
    max_sortino = population.max(metric=Metrics.SORTINO_RATIO)

    # Test the portfolios on the test period
    for portfolio in population.get_portfolios_by_tag(tag='markowitz'):
        new_portfolio = Portfolio(weights=portfolio.weights,
                                  fitness_type=FitnessType.MEAN_STD,
                                  assets=assets_19_20,
                                  name=portfolio.name,
                                  tag='markowitz_19_20')
        population.add(new_portfolio)

    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN, color_scale='name')

    # Mean-semivariance
    population = Population()
    optimal_weights = mean_semivariance_optimization(mu=assets.mu,
                                                     returns=assets.returns,
                                                     returns_target=assets.mu,
                                                     investment_type=InvestmentType.FULLY_INVESTED,
                                                     weight_bounds=(0, None),
                                                     population_size=100)
    for i, weights in enumerate(optimal_weights):
        portfolio = Portfolio(weights=weights,
                              fitness_type=FitnessType.MEAN_STD,
                              assets=assets,
                              name=str(i),
                              tag='semivariance')
        population.add(portfolio)

    # Plot
    population.plot(x=Metrics.ANNUALIZED_DOWNSIDE_STD, y=Metrics.ANNUALIZED_MEAN, color_scale='sharpe_ratio')
