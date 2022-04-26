import numpy as np
import itertools

from portfolio_optimization.assets import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *

__all__ = ['pre_selection']


def pre_selection(assets: Assets, k: int) -> list[str]:
    """
    Assets Preselection Process 2 from "Large-Scale Portfolio Optimization Using Multiobjective
    Evolutionary Algorithms and Preselection Methods" by B.Y. Qu and Q.Zhou - 2017.

    Good single asset (high return and low risk) is likely to contribute to the final optimized portfolio.
    Each asset is considered as a portfolio and these assets are ranked using the nondomination sorting method.
    The selection is based on the ranks assigned to each asset until the number of selected assets reaches
    the user-defined number.

    Considering only the risk and return of individual asset is insufficient because a pair of negatively correlated
    assets has the potential to reduce the risk. Therefore, negatively correlated pairs of assets are also considered.

    We find the portfolio with minimum variance obtained by the pair negatively correlated assets
    and includes it in the nondomination sorting process.

    ptf_variance = 𝜎1^2 𝑤1^2 + 𝜎2^2 𝑤2^2 + 2 𝜎12 𝑤1 𝑤2 (1)
    with 𝑤1 + 𝑤2 = 1

    To find the minimum we substitute 𝑤2 = 1 - 𝑤1 in (1) and different with respect to 𝑤1 and set to zero.
    By solving the obtained equation, we get:
    𝑤1 = (𝜎2^2 - 𝜎12) / (𝜎1^2 + 𝜎2^2 - 2 𝜎12)
    𝑤2 = 1 - 𝑤1

    :param assets: Assets class from which we will perform the pre-selection process
    :param k: minimum number of assets to pre-select. If k is reached before the end of the current front, we will
    the remaining assets of the current front. We do that because all assets in the same front have same rank.
    """

    if k >= assets.asset_nb:
        return assets.names

    # Build a population of portfolio
    population = Population()

    # Add single assets
    for i in range(assets.asset_nb):
        weights = np.zeros(assets.asset_nb)
        weights[i] = 1
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets)
        population.add(portfolio)

    # Add negatively correlated pairs with minimum variance
    for i, j in itertools.combinations(range(assets.asset_nb), 2):
        cov = assets.cov[i, j]
        if cov > 0:
            continue
        var1 = assets.cov[i, i]
        var2 = assets.cov[j, j]
        weights = np.zeros(assets.asset_nb)
        weights[i] = (var2 - cov) / (var1 + var2 - 2 * cov)
        weights[j] = 1 - weights[i]
        portfolio = Portfolio(weights=weights, fitness_type=FitnessType.MEAN_STD, assets=assets)
        population.add(portfolio)

    new_assets_idx = set()
    i = 0
    while i < len(population.fronts) and len(new_assets_idx) < k:
        front = population.fronts[i]
        for idx in front:
            new_assets_idx.update(population.portfolios[idx].assets_index)
        i += 1

    new_assets_names = list(assets.prices.columns[list(new_assets_idx)])
    return new_assets_names
