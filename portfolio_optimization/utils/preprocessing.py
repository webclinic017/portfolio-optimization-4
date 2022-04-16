import logging
from typing import Union
import numpy as np
import pandas as pd

logger = logging.getLogger('portfolio_optimization.preprocessing')

__all__ = ['preprocessing']


def preprocessing(prices: pd.DataFrame,
                  asset_missing_threshold: float = 0.05,
                  dates_missing_threshold: float = 0.1,
                  to_array: bool = True) -> Union[np.ndarray, pd.DataFrame]:
    """
    1) Remove Assets (columns) with missing value (nan) above asset_missing_threshold
    2) Remove Dates (rows) with missing value (nan) above dates_missing_threshold
    3) Compute simple returns (R1 = S1/S0 - 1)
        --> simple returns leads to a better estimate of the efficient frontier than log returns
    4) Convert to numpy array of shape (assets, dates)
    """
    assert asset_missing_threshold < 1
    assert dates_missing_threshold < 1

    n, m = prices.shape
    # Remove assets with missing prices above threshold
    count_nan = prices.isna().sum(axis=0)
    to_drop = count_nan[count_nan > n * asset_missing_threshold].index
    prices.drop(to_drop, axis=1, inplace=True)
    logger.info(f'{len(to_drop)} assets removed because they had more '
                f'that {asset_missing_threshold * 100}% prices missing')
    # Remove dates with missing prices above threshold
    count_nan = prices.isna().sum(axis=1)
    to_drop = count_nan[count_nan > m * dates_missing_threshold].index
    prices.drop(to_drop, axis=0, inplace=True)
    logger.info(f'{len(to_drop)} dates removed because they had more '
                f'that {dates_missing_threshold * 100}% prices missing')
    # Forward fill missing values
    prices.fillna(method='ffill', inplace=True)
    # Backward fill missing values to the start
    prices.fillna(method='bfill', inplace=True)
    returns = prices.pct_change()[1:]
    if to_array:
        return returns.to_numpy().T
    return returns
