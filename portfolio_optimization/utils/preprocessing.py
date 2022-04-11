import logging
import pandas as pd

logger = logging.getLogger('portfolio_optimization.preprocessing')

__all__ = ['preprocessing']


def preprocessing(df: pd.DataFrame,
                  asset_missing_threshold: float = 0.05,
                  dates_missing_threshold: float = 0.1) -> pd.DataFrame:
    """
    1) Remove Assets (columns) with missing value (nan) above asset_missing_threshold
    2) Remove Dates (rows) with missing value (nan) above dates_missing_threshold
    3) Compute simple returns (R1 = S1/S0 - 1)
        --> simple returns leads to a better estimate of the efficient frontier than log returns

    """
    assert asset_missing_threshold < 1
    assert dates_missing_threshold < 1

    n, m = df.shape
    # Remove assets with missing prices above threshold
    count_nan = df.isna().sum(axis=0)
    to_drop = count_nan[count_nan > n * asset_missing_threshold].index
    df.drop(to_drop, axis=1, inplace=True)
    logger.info(f'{len(to_drop)} assets removed because they had more '
                f'that {asset_missing_threshold * 100}% prices missing')
    # Remove dates with missing prices above threshold
    count_nan = df.isna().sum(axis=1)
    to_drop = count_nan[count_nan > m * dates_missing_threshold].index
    df.drop(to_drop, axis=0, inplace=True)
    logger.info(f'{len(to_drop)} dates removed because they had more '
                f'that {dates_missing_threshold * 100}% prices missing')
    # Forward fill missing values
    df.fillna(method='ffill', inplace=True)
    # Backward fill missing values to the start
    df.fillna(method='bfill', inplace=True)
    return df.pct_change()[1:]
