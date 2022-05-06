import logging
from typing import Optional
import datetime as dt
import pandas as pd
import numpy as np

from portfolio_optimization.bloomberg.loader import *

pd.options.plotting.backend = "plotly"

__all__ = ['Assets']

logger = logging.getLogger('portfolio_optimization.assets')


class Assets:
    def __init__(self,
                 start_date: dt.date,
                 end_date: Optional[dt.date] = None,
                 asset_missing_threshold: Optional[float] = 0.1,
                 dates_missing_threshold: Optional[float] = 0.1,
                 names_to_keep: Optional[list[str]] = None,
                 random_selection: Optional[int] = None,
                 name: Optional[str] = 'assets'):

        """
        :param start_date: starting date
        :param end_date: ending date
        :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
        :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
        :param names_to_keep: asset names to keep in the final DataFrame
        :param random_selection: number of assets to randomly keep in the final DataFrame
        :param name: name of the Assets class
        """

        if not 0 < asset_missing_threshold < 1:
            raise ValueError(f'asset_missing_threshold has to be between 0 and 1')
        if not 0 < dates_missing_threshold < 1:
            raise ValueError(f'dates_missing_threshold has to be between 0 and 1')

        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.prices = load_bloomberg_prices(date_from=self.start_date,
                                            date_to=self.end_date,
                                            names_to_keep=names_to_keep,
                                            random_selection=random_selection)
        self.asset_missing_threshold = asset_missing_threshold
        self.dates_missing_threshold = dates_missing_threshold
        self._preprocessing()
        self._returns = None
        self._cum_returns = None
        self._mu = None
        self._cov = None
        self._corr = None

    def _preprocessing(self):
        """
        1) Remove Assets (columns) with missing dates (nan) above dates_missing_threshold
        2) Remove Dates (rows) with missing assets (nan) above asset_missing_threshold
        """
        n, m = self.prices.shape

        # Remove assets with missing prices above threshold
        count_nan = self.prices.isna().sum(axis=0)
        to_drop = count_nan[count_nan > n * self.dates_missing_threshold].index
        if len(to_drop) > 0:
            self.prices.drop(to_drop, axis=1, inplace=True)
            logger.info(f'{len(to_drop)} assets removed because they had more '
                        f'that {self.dates_missing_threshold * 100}% prices missing')

        # Remove dates with missing prices above threshold
        count_nan = self.prices.isna().sum(axis=1)
        to_drop = count_nan[count_nan > m * self.asset_missing_threshold].index
        if len(to_drop) > 0:
            self.prices.drop(to_drop, axis=0, inplace=True)
            logger.info(f'{len(to_drop)} dates removed because they had more '
                        f'that {self.asset_missing_threshold * 100}% prices missing')
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
    def corr(self):
        if self._corr is None:
            self._corr = np.corrcoef(self.returns)
        return self._corr

    @property
    def asset_nb(self):
        return self.returns.shape[0]

    @property
    def date_nb(self):
        return self.returns.shape[1]

    @property
    def names(self):
        return np.array(self.prices.columns)

    def plot(self, idx=slice(None)):
        fig = self.cum_returns.iloc[:, idx].plot(title='Prices')
        fig.show()

    def __str__(self):
        return f'Assets <{self.name}  - {self.asset_nb} assets - {self.date_nb} dates>'

    def __repr__(self):
        return str(self)
