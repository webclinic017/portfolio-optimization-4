import logging
import datetime as dt
import pandas as pd
import numpy as np

from portfolio_optimization.bloomberg.loader import *
pd.options.plotting.backend = "plotly"


__all__ = ['Assets']

logger = logging.getLogger('portfolio_optimization.assets')


class Assets:
    def __init__(self,
                 date_from: dt.date,
                 date_to: dt.date = None,
                 asset_missing_threshold: float = 0.05,
                 dates_missing_threshold: float = 0.1,
                 names_to_keep: list[str] = None):
        assert asset_missing_threshold < 1
        assert dates_missing_threshold < 1
        self.date_from = date_from
        self.date_to = date_to
        self.prices = load_bloomberg_prices(date_from=self.date_from, date_to=self.date_to, names_to_keep=names_to_keep)
        self.asset_missing_threshold = asset_missing_threshold
        self.dates_missing_threshold = dates_missing_threshold
        self._preprocessing()
        self._returns = None
        self._cum_returns = None
        self._mu = None
        self._cov = None

    def _preprocessing(self):
        """
        1) Remove Assets (columns) with missing value (nan) above asset_missing_threshold
        2) Remove Dates (rows) with missing value (nan) above dates_missing_threshold
        """
        n, m = self.prices.shape
        # Remove assets with missing prices above threshold
        count_nan = self.prices.isna().sum(axis=0)
        to_drop = count_nan[count_nan > n * self.asset_missing_threshold].index
        self.prices.drop(to_drop, axis=1, inplace=True)
        logger.info(f'{len(to_drop)} assets removed because they had more '
                    f'that {self.asset_missing_threshold * 100}% prices missing')
        # Remove dates with missing prices above threshold
        count_nan = self.prices.isna().sum(axis=1)
        to_drop = count_nan[count_nan > m * self.dates_missing_threshold].index
        self.prices.drop(to_drop, axis=0, inplace=True)
        logger.info(f'{len(to_drop)} dates removed because they had more '
                    f'that {self.dates_missing_threshold * 100}% prices missing')
        # Forward fill missing values
        self.prices.fillna(method='ffill', inplace=True)
        # Backward fill missing values to the start
        self.prices.fillna(method='bfill', inplace=True)

    @property
    def returns(self):
        """
        Compute simple returns from prices (R1 = S1/S0 - 1)
            --> simple returns leads to a better estimate of the efficient frontier than log returns
        And convert to numpy array of shape (assets, dates)
        """
        if self._returns is None:
            self._returns = self.prices.pct_change()[1:].to_numpy().T
        return self._returns

    @property
    def cum_returns(self):
        """
        Compute the cumulative returns (1+R1)*(1+R2)*(1+R3)...  = S1/S0 - 1)
        It's like rebasing prices to 1
        """
        if self._cum_returns is None:
            df_returns = self.prices.pct_change()[1:]
            self._cum_returns = (df_returns + 1).cumprod()
        return self._cum_returns

    @property
    def mu(self):
        if self._mu is None:
            self._mu = np.mean(self.returns, axis=1)
        return self._mu

    @property
    def cov(self):
        if self._cov is None:
            self._cov = np.cov(self.returns)
        return self._cov

    @property
    def asset_nb(self):
        return self.returns.shape[0]

    @property
    def date_nb(self):
        return self.returns.shape[1]

    @property
    def names(self):
        return list(self.prices.columns)

    def plot(self, idx=slice(None)):
        fig = self.cum_returns.iloc[:, idx].plot(title='Prices')
        fig.show()

    def __str__(self):
        return f'Assets (asset number: {self.asset_nb}, date number: {self.date_nb})'

    def __repr__(self):
        return str(self)
