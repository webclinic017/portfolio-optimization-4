import datetime as dt
import numpy as np
import cvxpy as cp

from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.utils.pre_seclection import *

np.random.seed(150)


def markowitz_optimization(mu: np.matrix, cov: np.matrix, population_size: int) -> np.array:
    """
    Markowitz optimization:
    Constraints: No-short selling and portfolio invested at 100%
    """
    w = cp.Variable(len(mu))
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, cov)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk),
                      [cp.sum(w) == 1,
                       w >= 0])

    portfolios_mu = []
    portfolios_std = []
    weights = []
    for value in np.logspace(-2, 3, num=population_size):
        gamma.value = value
        prob.solve()
        portfolios_mu.append(ret.value)
        portfolios_std.append(np.sqrt(risk.value))
        weights.append(w.value)

    return np.array(weights)


def run():
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
    optimal_weights = markowitz_optimization(mu=assets.mu, cov=assets.cov, population_size=20)
    for i, weights in enumerate(optimal_weights):
        portfolio = Portfolio(weights=weights,
                              fitness_type=FitnessType.MEAN_STD,
                              assets=assets,
                              name=str(i),
                              tag='markowitz')
        population.append(portfolio)

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN, color_scale='sharp_ratio')
    max_sortino=population.max(metric=Metrics.SORTINO_RATIO)

    # Test the portfolios on the test period
    for portfolio in population.get_portfolios(tag='markowitz'):
        new_portfolio = Portfolio(weights=portfolio.weights,
                                  fitness_type=FitnessType.MEAN_STD,
                                  assets=assets_19_20,
                                  name=portfolio.name,
                                  tag='markowitz_19_20')
        population.append(new_portfolio)

    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN, color_scale='name')
