import itertools
import logging
from typing import Optional
import datetime as dt
import numpy as np

from portfolio_optimization.assets import *
from portfolio_optimization.population import *
from portfolio_optimization.portfolio import *

__all__ = ['pre_selection',
           'load_train_test_assets',
           'load_assets_with_preselection']

logger = logging.getLogger('portfolio_optimization.utils.assets')


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
        return list(assets.names)

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


def load_assets_with_preselection(start_date: dt.date,
                                  end_date: Optional[dt.date] = None,
                                  asset_missing_threshold: Optional[float] = 0.1,
                                  dates_missing_threshold: Optional[float] = 0.1,
                                  names_to_keep: Optional[list[str]] = None,
                                  random_selection: Optional[int] = None,
                                  pre_selection_number: Optional[int] = None,
                                  name: Optional[str] = 'assets') -> Assets:
    """
    Load Assets form multiple periods
    :param start_date: starting date
    :param end_date: ending date
    :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
    :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
    :param names_to_keep: asset names to keep in the final DataFrame
    :param random_selection: number of assets to randomly keep in the final DataFrame
    :param pre_selection_number: number of assets to pre-select using the Assets Preselection Process
    :param name: name of the Assets class

    """
    assets = Assets(start_date=start_date,
                    end_date=end_date,
                    asset_missing_threshold=asset_missing_threshold,
                    dates_missing_threshold=dates_missing_threshold,
                    names_to_keep=names_to_keep,
                    random_selection=random_selection,
                    name=name)

    if pre_selection_number is not None:
        assets_pre_selected = pre_selection(assets=assets, k=pre_selection_number)
        assets = Assets(start_date=start_date,
                        end_date=end_date,
                        asset_missing_threshold=asset_missing_threshold,
                        dates_missing_threshold=dates_missing_threshold,
                        names_to_keep=assets_pre_selected,
                        name=name)

    return assets


def load_train_test_assets(train_period: (dt.date, dt.date),
                           test_period: (dt.date, dt.date),
                           asset_missing_threshold: Optional[float] = 0.1,
                           dates_missing_threshold: Optional[float] = 0.1,
                           names_to_keep: Optional[list[str]] = None,
                           random_selection: Optional[int] = None,
                           pre_selection_number: Optional[int] = None) -> (Assets, Assets):
    """
    Load Assets form multiple periods
    :param train_period
    :param test_period
    :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
    :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
    :param names_to_keep: asset names to keep in the final DataFrame
    :param random_selection: number of assets to randomly keep in the final DataFrame
    :param pre_selection_number: number of assets to pre-select using the Assets Preselection Process
    :return a tuple of train Assets and test Assets
    """
    train_start, train_end = train_period
    test_start, test_end = test_period
    train_name = f'{train_start}-{train_end}'
    test_name = f'{test_start}-{test_end}'

    if train_start >= train_end or test_start >= test_end:
        raise ValueError(f'Periods are incorrect')

    if train_start < test_start < train_end or train_start < test_end < train_end:
        logger.warning(f'Train and Test periods are overlapping')

    train_assets = Assets(start_date=train_start,
                          end_date=train_end,
                          asset_missing_threshold=asset_missing_threshold,
                          dates_missing_threshold=dates_missing_threshold,
                          names_to_keep=names_to_keep,
                          random_selection=random_selection,
                          name=train_name)

    if pre_selection_number is not None:
        assets_pre_selected = pre_selection(assets=train_assets, k=pre_selection_number)
        train_assets = Assets(start_date=train_start,
                              end_date=train_end,
                              asset_missing_threshold=asset_missing_threshold,
                              dates_missing_threshold=dates_missing_threshold,
                              names_to_keep=assets_pre_selected,
                              name=train_name)

    test_assets = Assets(start_date=test_start,
                         end_date=test_end,
                         asset_missing_threshold=asset_missing_threshold,
                         dates_missing_threshold=dates_missing_threshold,
                         names_to_keep=train_assets.names,
                         name=test_name)

    # Ensure than train_assets and test_assets contains the same assets
    if set(train_assets.names) != set(test_assets.names):
        names = [name for name in train_assets.names if name in test_assets.names]
        if set(train_assets.names) != set(names):
            train_assets = Assets(start_date=train_start,
                                  end_date=train_end,
                                  asset_missing_threshold=asset_missing_threshold,
                                  dates_missing_threshold=dates_missing_threshold,
                                  names_to_keep=names,
                                  name=train_name)

        if set(test_assets.names) != set(names):
            test_assets = Assets(start_date=test_start,
                                 end_date=test_end,
                                 asset_missing_threshold=asset_missing_threshold,
                                 dates_missing_threshold=dates_missing_threshold,
                                 names_to_keep=names,
                                 name=test_name)
    if set(train_assets.names) != set(test_assets.names):
        raise ValueError(f'Unable to generate train and test period with identical asset names')

    return train_assets, test_assets
