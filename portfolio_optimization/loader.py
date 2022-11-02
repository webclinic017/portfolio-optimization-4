import logging
import datetime as dt
import numpy as np
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.population import *
from portfolio_optimization.portfolio import *

__all__ = ['pre_selection',
           'load_train_test_assets',
           'load_assets']

logger = logging.getLogger('portfolio_optimization.loader')


def pre_selection(assets: Assets,
                  k: int,
                  correlation_threshold: float = 0) -> list[str]:
    """
    Assets Preselection Process 2 from "Large-Scale Portfolio Optimization Using Multi-objective
    Evolutionary Algorithms and Preselection Methods" by B.Y. Qu and Q.Zhou - 2017.

    Good single asset (high return and low risk) is likely to contribute to the final optimized portfolio.
    Each asset is considered as a portfolio and these assets are ranked using the non-domination sorting method.
    The selection is based on the ranks assigned to each asset until the number of selected assets reaches
    the user-defined number.

    Considering only the risk and return of individual asset is insufficient because a pair of negatively correlated
    assets has the potential to reduce the risk. Therefore, negatively correlated pairs of assets are also considered.

    We find the portfolio with minimum variance obtained by the pair of negatively correlated assets. We then includes
    it in the non-domination sorting process.

    ptf_variance = 𝜎1^2 𝑤1^2 + 𝜎2^2 𝑤2^2 + 2 𝜎12 𝑤1 𝑤2 (1)
    with 𝑤1 + 𝑤2 = 1

    To find the minimum we substitute 𝑤2 = 1 - 𝑤1 in (1) and differentiate with respect to 𝑤1 and set to zero.
    By solving the obtained equation, we get:
    𝑤1 = (𝜎2^2 - 𝜎12) / (𝜎1^2 + 𝜎2^2 - 2 𝜎12)
    𝑤2 = 1 - 𝑤1
    :param assets: Assets class from which we will perform the pre-selection process
    :param k: minimum number of assets to pre-select. If k is reached before the end of the current front, we will
           the remaining assets of the current front. We do that because all assets in the same front have same rank
    :param correlation_threshold: asset pair with a correlation below correlation_threshold are included in the
           non-domination sorting, default is 0
    """

    if k >= assets.asset_nb:
        return list(assets.names)

    # Build a population of portfolio
    population = Population()
    # Add single assets
    for i in range(assets.asset_nb):
        weights = np.zeros(assets.asset_nb)
        weights[i] = 1
        portfolio = Portfolio(weights=weights, fitness_metrics=FitnessType.MEAN_STD, assets=assets)
        population.append(portfolio)

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
                portfolio = Portfolio(weights=weights, assets=assets)
                population.append(portfolio)

    new_assets_idx = set()
    i = 0
    while i < len(population.fronts) and len(new_assets_idx) < k:
        front = population.fronts[i]
        for idx in front:
            new_assets_idx.update(population[idx].assets_index)
        i += 1

    new_assets_names = list(assets.prices.columns[list(new_assets_idx)])

    return new_assets_names


def load_assets(prices: pd.DataFrame,
                start_date: dt.date | None = None,
                end_date: dt.date | None = None,
                asset_missing_threshold: float | None = None,
                dates_missing_threshold: float | None = None,
                names_to_keep: list[str] | np.ndarray | None = None,
                random_selection: int | None = None,
                removal_correlation: float | None = None,
                pre_selection_number: int | None = None,
                pre_selection_correlation: float | None = None,
                name: str | None = None,
                verbose: bool = True) -> Assets:
    """
    Load Assets form multiple periods
    :param prices: DataFrame of asset prices. Index has to be DateTime and columns names are the assets names
    :param start_date: starting date
    :param end_date: ending date
    :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
    :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
    :param names_to_keep: asset names to keep in the final DataFrame
    :param random_selection: number of assets to randomly keep in the final DataFrame
    :param removal_correlation: when two assets have a correlation above this threshold,
            we keep the asset with higher returns
    :param pre_selection_number: number of assets to pre-select using the Assets Preselection Process 2 from
           "Large-Scale Portfolio Optimization Using Multi-objective Evolutionary Algorithms and Preselection
           Methods" by B.Y. Qu and Q.Zhou - 2017
    :param pre_selection_correlation: asset pair with a correlation below this threshold are included in the
           non-domination sorting of the pre-selection method
    :param name: name of the Assets class
    :param verbose: True to print logging info
    """

    assets = Assets(name=name,
                    prices=prices,
                    start_date=start_date,
                    end_date=end_date,
                    asset_missing_threshold=asset_missing_threshold,
                    dates_missing_threshold=dates_missing_threshold,
                    names_to_keep=names_to_keep,
                    random_selection=random_selection,
                    correlation_threshold=removal_correlation,
                    verbose=verbose)

    if pre_selection_number is not None or pre_selection_correlation is not None:
        if pre_selection_number is None or pre_selection_correlation is None:
            raise ValueError('pre_selection_number and pre_selection_correlation have to be both provided '
                             'or both None')
        new_assets_names = pre_selection(assets=assets,
                                         k=pre_selection_number,
                                         correlation_threshold=pre_selection_correlation)
        # Reloading Assets after removing assets discarded form the pre-selection process
        assets = Assets(name=name,
                        prices=prices,
                        start_date=start_date,
                        end_date=end_date,
                        asset_missing_threshold=asset_missing_threshold,
                        dates_missing_threshold=dates_missing_threshold,
                        correlation_threshold=None,
                        names_to_keep=new_assets_names,
                        verbose=verbose)

    return assets


def load_train_test_assets(prices: pd.DataFrame,
                           train_period: (dt.date, dt.date),
                           test_period: (dt.date, dt.date),
                           asset_missing_threshold: float | None = None,
                           dates_missing_threshold: float | None = None,
                           names_to_keep: list[str] | np.ndarray | None = None,
                           random_selection: int | None = None,
                           removal_correlation: float | None = None,
                           pre_selection_number: int | None = None,
                           pre_selection_correlation: float | None = None,
                           verbose: bool = True) -> tuple[Assets, Assets]:
    """
    Load Assets form multiple periods
    :param prices: DataFrame of asset prices. Index has to be DateTime and columns names are the assets names
    :param train_period
    :param test_period
    :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
    :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
    :param names_to_keep: asset names to keep in the final DataFrame
    :param random_selection: number of assets to randomly keep in the final DataFrame
    :param removal_correlation: when two assets have a correlation above this threshold,
            we keep the asset with higher returns
    :param pre_selection_number: number of assets to pre-select using the Assets Preselection Process 2 from
           "Large-Scale Portfolio Optimization Using Multi-objective Evolutionary Algorithms and Preselection
           Methods" by B.Y. Qu and Q.Zhou - 2017
    :param pre_selection_correlation: asset pair with a correlation below this threshold are included in the
           non-domination sorting of the pre-selection method
    :param verbose: True to print logging info
    :return a tuple of train Assets and test Assets
    """
    train_start, train_end = train_period
    test_start, test_end = test_period
    train_name = f'train_{train_start}-{train_end}'
    test_name = f'test_{test_start}-{test_end}'

    if train_start >= train_end or test_start >= test_end:
        raise ValueError(f'Periods are incorrect')

    if train_start < test_start < train_end or train_start < test_end < train_end:
        logger.warning(f'Train and Test periods are overlapping')
    # Loading Train Assets from train_start to train_end
    train_assets = Assets(prices=prices,
                          start_date=train_start,
                          end_date=train_end,
                          asset_missing_threshold=asset_missing_threshold,
                          dates_missing_threshold=dates_missing_threshold,
                          correlation_threshold=removal_correlation,
                          names_to_keep=names_to_keep,
                          random_selection=random_selection,
                          name=train_name,
                          verbose=verbose)

    if pre_selection_number is not None or pre_selection_correlation is not None:
        if pre_selection_number is None or pre_selection_correlation is None:
            raise ValueError('pre_selection_number and pre_selection_correlation have to be both provided '
                             'or both None')
        new_assets_names = pre_selection(assets=train_assets,
                                         k=pre_selection_number,
                                         correlation_threshold=pre_selection_correlation)
        # Reloading Train Assets after removing assets discarded form the pre-selection process
        train_assets = Assets(prices=prices,
                              start_date=train_start,
                              end_date=train_end,
                              asset_missing_threshold=asset_missing_threshold,
                              dates_missing_threshold=dates_missing_threshold,
                              correlation_threshold=None,
                              names_to_keep=new_assets_names,
                              name=train_name,
                              verbose=verbose)
    # Loading Test Assets
    test_assets = Assets(prices=prices,
                         start_date=test_start,
                         end_date=test_end,
                         asset_missing_threshold=asset_missing_threshold,
                         dates_missing_threshold=dates_missing_threshold,
                         correlation_threshold=None,
                         names_to_keep=train_assets.names,
                         name=test_name,
                         verbose=verbose)

    # Ensure than train_assets and test_assets contains the same assets
    if set(train_assets.names) != set(test_assets.names):
        names = [name for name in train_assets.names if name in test_assets.names]
        if set(train_assets.names) != set(names):
            train_assets.keep_assets(assets_to_keep=list(names))

        if set(test_assets.names) != set(names):
            test_assets.keep_assets(assets_to_keep=list(names))

    if set(train_assets.names) != set(test_assets.names):
        raise ValueError(f'Unable to generate train and test period with identical asset names')

    return train_assets, test_assets
