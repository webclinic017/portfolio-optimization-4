import datetime as dt
import numpy as np
import cvxpy as cp

from portfolio_optimization.utils.tools import *
from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.utils.pre_seclection import *


np.random.seed(150)


def portfolio_with_optimal_sharpe(mu: np.matrix,
                                  cov: np.matrix,
                                  portfolios_mu: np.array,
                                  portfolios_std: np.array) -> np.array:
    # Compute the second degree polynomial of the pareto front
    m1 = np.polyfit(portfolios_mu, portfolios_std, 2)
    target_ret = np.sqrt(m1[2] / m1[0])
    # compute the portfolio with optimal sharpe ratio
    w = cp.Variable(len(mu))
    ret = mu.T @ w
    risk = cp.quad_form(w, cov)
    prob = cp.Problem(cp.Minimize(risk),
                      [ret == target_ret,
                       cp.sum(w) == 1,
                       w >= 0])
    prob.solve()
    weights = w.value
    return weights


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

    optimal_sharpe_weights = portfolio_with_optimal_sharpe(mu=mu,
                                                           cov=cov,
                                                           portfolios_mu=np.array(portfolios_mu),
                                                           portfolios_std=np.array(portfolios_std))

    return np.array(weights), optimal_sharpe_weights



def run():
    assets = Assets(date_from=dt.date(2019, 1, 1))
    assets_pre_selected = pre_selection(assets=assets, k=200)
    assets = Assets(date_from=dt.date(2019, 1, 1), names_to_keep=assets_pre_selected[:150])

    population = Population()

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

    # Pareto optimal portfolios
    optimal_weights, optimal_sharpe_weights = markowitz_optimization(mu=assets.mu, cov=assets.cov, population_size=100)
    for weights in optimal_weights:
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets, tag='markowitz')
        population.append(portfolio)
    portfolio = Portfolio(weights=optimal_sharpe_weights, fitness_type=FitnessType.MEAN_STD, assets=assets,
                          tag='optimal_sharpe')
    population.append(portfolio)

    # Plot
    population.plot(x='annualized_std', y='annualized_mu', color_scale='sharp_ratio')
