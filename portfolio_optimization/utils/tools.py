import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

__all__ = ['prices_rebased',
           'portfolio_returns',
           'rand_weights',
           'rand_weights_dirichlet',
           'walk_forward',
           'load_prices',
           'args_names',
           'clean',
           'clean_locals']


def clean(x: float | list | np.ndarray | None,
          dtype: type | str | None = None) -> float | np.ndarray | None:
    r"""
    Clean function arguments by converting list into ndarray and leaving other type unchanged
    """
    if isinstance(x, list):
        return np.array(x, dtype=dtype)
    return x


def clean_locals(f_locals: dict, /) -> dict:
    r"""
    Clean locals by removing self
    """
    return {k: v for k, v in f_locals.items() if k != 'self'}


def prices_rebased(returns: np.array) -> np.array:
    r"""
    Convert a return series into a compounded return stating from 1.
    """
    p = [1 + returns[0]]
    for i in range(1, len(returns)):
        p.append(p[-1] * (1 + returns[i]))
    return np.array(p)


def portfolio_returns(asset_returns: np.ndarray, weights: np.array) -> np.array:
    r"""
    Compute the portfolio returns from its assets returns and weights
    """
    n, m = asset_returns.shape
    returns = np.zeros(m)
    for i in range(n):
        returns += asset_returns[i] * weights[i]
    return returns


def rand_weights(n: int, zeros: int = 0) -> np.array:
    r"""
    Produces n random weights that sum to 1 (non-uniform distribution over the simplex)
    """
    k = np.random.rand(n)
    if zeros > 0:
        zeros_idx = np.random.choice(n, zeros, replace=False)
        k[zeros_idx] = 0
    return k / sum(k)


def rand_weights_dirichlet(n: int) -> np.array:
    r"""
    Produces n random weights that sum to 1 with uniform distribution over the simplex
    """
    return np.random.dirichlet(np.ones(n))


def walk_forward(start_date: dt.date,
                 end_date: dt.date,
                 train_duration: int,
                 test_duration: int,
                 full_period: bool = True) -> tuple[tuple[dt.date, dt.date], tuple[dt.date, dt.date]]:
    r"""
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
    r"""
    Read prices csv and return a DataFrame

    Parameters
    ----------
    file: Path | str
          The path of the prices csv file

    Returns
    -------
    Prices DataFrame
    """
    df = pd.read_csv(file, sep=',', index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return df


def args_names(func: object) -> list[str]:
    r"""
    Returns the argument names of a function
    """
    return [v for v in func.__code__.co_varnames[:func.__code__.co_argcount] if v != 'self']
