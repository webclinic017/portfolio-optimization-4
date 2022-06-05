import logging
from pathlib import Path
from typing import Optional, Union
import datetime as dt
import numpy as np

from portfolio_optimization.assets import *
from portfolio_optimization.meta import FitnessType
from portfolio_optimization.population import *
from portfolio_optimization.portfolio import *

__all__ = ['remove_highly_correlated_assets',
           'pre_selection',
           'load_train_test_assets',
           'load_assets']

logger = logging.getLogger('portfolio_optimization.utils.assets')


def remove_highly_correlated_assets(assets: Assets, correlation_threshold: float = 0.99) -> list[str]:
    """
    When two assets have a correlation above correlation_threshold, we keep the asset with higher returns.
    Highly correlated assets increase calculus overhead and can cause matrix calculus errors without adding
    significant information.

    :param assets: Assets class
    :param correlation_threshold: correlation threshold
    :return: asset names
    """
    if not -1 <= correlation_threshold <= 1:
        raise ValueError(f'correlation_threshold has to be between -1 and 1')

    n = assets.asset_nb
    to_remove = set()
    for i in range(n - 1):
        for j in range(i + 1, n):
            if assets.corr[i, j] > correlation_threshold:
                if i not in to_remove and j not in to_remove:
                    if assets.mu[i] < assets.mu[j]:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
    logger.info(f'{len(to_remove)} assets removed with a correlation above {correlation_threshold}')
    new_assets_names = list(np.delete(assets.names, list(to_remove)))
    return new_assets_names


def pre_selection(assets: Assets,
                  k: int,
                  correlation_threshold: float = 0) -> list[str]:
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
    :param correlation_threshold: asset pair with a correlation below correlation_threshold are included in the
           nondomination sorting, default is 0
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
    n = assets.asset_nb
    for i in range(n - 1):
        for j in range(i + 1, n):
            if assets.corr[i, j] < correlation_threshold:
                cov = assets.cov[i, j]
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


def load_assets(prices_file: Union[Path, str],
                start_date: Optional[dt.date] = None,
                end_date: Optional[dt.date] = None,
                asset_missing_threshold: float = 0.1,
                dates_missing_threshold: float = 0.1,
                names_to_keep: Optional[list[str]] = None,
                random_selection: Optional[int] = None,
                correlation_threshold_removal: float = 0.99,
                pre_selection_number: Optional[int] = None,
                correlation_threshold_pre_selection: float = 0,
                name: Optional[str] = 'assets') -> Assets:
    """
    Load Assets form multiple periods
    :param prices_file: the path of the prices csv file
    :param start_date: starting date
    :param end_date: ending date
    :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
    :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
    :param names_to_keep: asset names to keep in the final DataFrame
    :param random_selection: number of assets to randomly keep in the final DataFrame
    :param correlation_threshold_removal: when two assets have a correlation above this threshold,
            we keep the asset with higher returns.
    :param pre_selection_number: number of assets to pre-select using the Assets Preselection Process
    :param correlation_threshold_pre_selection: asset pair with a correlation below this threshold are included in the
           nondomination sorting of the pre selection method.
    :param name: name of the Assets class
    """
    logger.info(f'Loading Assets from {start_date} to {end_date}')
    assets = Assets(prices_file=prices_file,
                    start_date=start_date,
                    end_date=end_date,
                    asset_missing_threshold=asset_missing_threshold,
                    dates_missing_threshold=dates_missing_threshold,
                    names_to_keep=names_to_keep,
                    random_selection=random_selection,
                    name=name)

    if correlation_threshold_removal is not None:
        new_assets_names = remove_highly_correlated_assets(assets=assets,
                                                           correlation_threshold=correlation_threshold_removal)
        logger.info(f'Reloading Assets after removing highly correlated assets')
        assets = Assets(prices_file=prices_file,
                        start_date=start_date,
                        end_date=end_date,
                        asset_missing_threshold=asset_missing_threshold,
                        dates_missing_threshold=dates_missing_threshold,
                        names_to_keep=new_assets_names,
                        name=name)

    if pre_selection_number is not None:
        new_assets_names = pre_selection(assets=assets,
                                         k=pre_selection_number,
                                         correlation_threshold=correlation_threshold_pre_selection)
        logger.info(f'Reloading Assets after removing assets discarded form the pre-selection process')
        assets = Assets(prices_file=prices_file,
                        start_date=start_date,
                        end_date=end_date,
                        asset_missing_threshold=asset_missing_threshold,
                        dates_missing_threshold=dates_missing_threshold,
                        names_to_keep=new_assets_names,
                        name=name)

    return assets


def load_train_test_assets(prices_file: Union[Path, str],
                           train_period: (dt.date, dt.date),
                           test_period: (dt.date, dt.date),
                           asset_missing_threshold: float = 0.1,
                           dates_missing_threshold: float = 0.1,
                           names_to_keep: Optional[list[str]] = None,
                           random_selection: Optional[int] = None,
                           correlation_threshold_removal: float = 0.99,
                           pre_selection_number: Optional[int] = None,
                           correlation_threshold_pre_selection: float = 0) -> (Assets, Assets):
    """
    Load Assets form multiple periods
    :param prices_file: the path of the prices csv file
    :param train_period
    :param test_period
    :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
    :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
    :param names_to_keep: asset names to keep in the final DataFrame
    :param random_selection: number of assets to randomly keep in the final DataFrame
    :param correlation_threshold_removal: when two assets have a correlation above this threshold,
            we keep the asset with higher returns.
    :param pre_selection_number: number of assets to pre-select using the Assets Preselection Process
    :param correlation_threshold_pre_selection: asset pair with a correlation below this threshold are included in the
           nondomination sorting of the pre selection method.
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
    logger.info(f'Loading Train Assets from {train_start} to {train_end}')
    train_assets = Assets(prices_file=prices_file,
                          start_date=train_start,
                          end_date=train_end,
                          asset_missing_threshold=asset_missing_threshold,
                          dates_missing_threshold=dates_missing_threshold,
                          names_to_keep=names_to_keep,
                          random_selection=random_selection,
                          name=train_name)

    if correlation_threshold_removal is not None:
        new_assets_names = remove_highly_correlated_assets(assets=train_assets,
                                                           correlation_threshold=correlation_threshold_removal)
        logger.info(f'Reloading Train Assets after removing highly correlated assets')
        train_assets = Assets(prices_file=prices_file,
                              start_date=train_start,
                              end_date=train_end,
                              asset_missing_threshold=asset_missing_threshold,
                              dates_missing_threshold=dates_missing_threshold,
                              names_to_keep=new_assets_names,
                              name=train_name)

    if pre_selection_number is not None:
        new_assets_names = pre_selection(assets=train_assets,
                                         k=pre_selection_number,
                                         correlation_threshold=correlation_threshold_pre_selection)
        logger.info(f'Reloading Train Assets after removing assets discarded form the pre-selection process')
        train_assets = Assets(prices_file=prices_file,
                              start_date=train_start,
                              end_date=train_end,
                              asset_missing_threshold=asset_missing_threshold,
                              dates_missing_threshold=dates_missing_threshold,
                              names_to_keep=new_assets_names,
                              name=train_name)
    logger.info(f'Loading Test Assets')
    test_assets = Assets(prices_file=prices_file,
                         start_date=test_start,
                         end_date=test_end,
                         asset_missing_threshold=asset_missing_threshold,
                         dates_missing_threshold=dates_missing_threshold,
                         names_to_keep=train_assets.names,
                         name=test_name)

    # Ensure than train_assets and test_assets contains the same assets
    if set(train_assets.names) != set(test_assets.names):
        names = [name for name in train_assets.names if name in test_assets.names]
        if set(train_assets.names) != set(names):
            logger.info(f'Reloading Train Assets to match Test Assets universe')
            train_assets = Assets(prices_file=prices_file,
                                  start_date=train_start,
                                  end_date=train_end,
                                  asset_missing_threshold=asset_missing_threshold,
                                  dates_missing_threshold=dates_missing_threshold,
                                  names_to_keep=names,
                                  name=train_name)

        if set(test_assets.names) != set(names):
            logger.info(f'Reloading Test Assets to match Train Assets universe')
            test_assets = Assets(prices_file=prices_file,
                                 start_date=test_start,
                                 end_date=test_end,
                                 asset_missing_threshold=asset_missing_threshold,
                                 dates_missing_threshold=dates_missing_threshold,
                                 names_to_keep=names,
                                 name=test_name)
    if set(train_assets.names) != set(test_assets.names):
        raise ValueError(f'Unable to generate train and test period with identical asset names')

    return train_assets, test_assets
