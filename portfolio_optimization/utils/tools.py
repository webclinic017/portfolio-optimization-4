import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

__all__ = ['prices_rebased',
           'portfolio_returns',
           'rand_weights',
           'rand_weights_dirichlet',
           'walk_forward',
           'load_prices']


def prices_rebased(returns: np.array) -> np.array:
    p = [1 + returns[0]]
    for i in range(1, len(returns)):
        p.append(p[-1] * (1 + returns[i]))
    return np.array(p)


def portfolio_returns(asset_returns: np.ndarray, weights: np.array) -> np.array:
    n, m = asset_returns.shape
    returns = np.zeros(m)
    for i in range(n):
        returns += asset_returns[i] * weights[i]
    return returns


def rand_weights(n: int, zeros: int = 0) -> np.array:
    """
    Produces n random weights that sum to 1 (non-uniform distribution over the simplex)
    """
    k = np.random.rand(n)
    if zeros > 0:
        zeros_idx = np.random.choice(n, zeros, replace=False)
        k[zeros_idx] = 0
    return k / sum(k)


def rand_weights_dirichlet(n: int) -> np.array:
    """
    Produces n random weights that sum to 1 with uniform distribution over the simplex
    """
    return np.random.dirichlet(np.ones(n))


def walk_forward(start_date: dt.date,
                 end_date: dt.date,
                 train_duration: int,
                 test_duration: int,
                 full_period: bool = True) -> tuple[tuple[dt.date, dt.date], tuple[dt.date, dt.date]]:
    """
    Yield train and test periods in a walk forward manner.

    The proprieties are:
        * The test periods are not overlapping
        * The test periods are directly following the train periods

    Example:
        0 --> Train
        1 -- Test

        00000111------
        ---00000111---
        ------00000111
    """
    train_start = start_date
    while True:
        train_end = train_start + dt.timedelta(days=train_duration)
        test_start = train_end
        test_end = test_start + dt.timedelta(days=test_duration)
        if test_end > end_date:
            if full_period or test_start >= end_date:
                return
            else:
                test_end = end_date
        yield (train_start, train_end), (test_start, test_end)
        train_start = train_start + dt.timedelta(days=test_duration)


def load_prices(file: Path | str) -> pd.DataFrame:
    """
    Read prices csv and return a DataFrame
    :param file: the path of the prices csv file
    """
    df = pd.read_csv(file, sep=',', index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return df
