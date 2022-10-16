import logging
import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functools import cached_property, cache

pd.options.plotting.backend = 'plotly'

__all__ = ['Assets']

logger = logging.getLogger('portfolio_optimization.assets')


class Assets:
    def __init__(self,
                 prices: pd.DataFrame,
                 name: str | None = None,
                 start_date: dt.date | None = None,
                 end_date: dt.date | None = None,
                 asset_missing_threshold: float | None = None,
                 dates_missing_threshold: float | None = None,
                 correlation_threshold: float | None = None,
                 names_to_keep: list[str] | np.ndarray | None = None,
                 random_selection: int | None = None,
                 verbose: bool = True):

        """
        Asset Class
        :param prices: DataFrame of asset prices. Index has to be DateTime and columns names are the assets names
        :param start_date: starting date
        :param end_date: ending date
        :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
        :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing
        :param correlation_threshold: when two assets have a correlation above this threshold,
                                      we keep the asset with higher returns
        :param names_to_keep: asset names to keep in the prices DataFrame
        :param random_selection: number of assets to randomly keep in the prices DataFrame
        :param name: name of the Assets class
        :param verbose: True to print logging info
        """
        if name is None:
            self.name = str(id(self))
        else:
            self.name = name
        self.verbose = verbose
        self.start_date = start_date
        self.end_date = end_date
        self.prices = prices.loc[start_date:end_date].copy()
        self._custom_expected_returns = None
        self._custom_expected_cov = None

        self._info(f'Loading Assets from {start_date} to {end_date}')

        if names_to_keep is not None:
            self.prices = self.prices[names_to_keep]

        if random_selection is not None:
            all_names = list(self.prices.columns)
            new_names = [all_names[i] for i in np.random.choice(len(all_names), random_selection, replace=False)]
            self.prices = self.prices[new_names]

        self._prices_check()
        self._preprocessing(asset_missing_threshold=asset_missing_threshold,
                            dates_missing_threshold=dates_missing_threshold)
        self._remove_highly_correlated_assets(correlation_threshold=correlation_threshold)

    def __str__(self):
        return f"<Assets, name: '{self.name}'>"

    def __repr__(self) -> str:
        return f"<Assets, name: '{self.name}'>"

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.dates), tuple(self.names)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Assets)
                and np.array_equal(self.dates, other.dates)
                and np.array_equal(self.names, other.names))

    @cached_property
    def dates(self) -> np.array:
        return np.array([date.date() for date in self.prices.index])

    @cached_property
    def names(self) -> np.ndarray:
        return np.array(self.prices.columns)

    @property
    def shape(self):
        return self.returns.shape

    @property
    def asset_nb(self):
        return self.shape[0]

    @property
    def date_nb(self):
        return self.shape[1]

    def _prices_check(self) -> None:
        """Sanity check of the prices DataFrame"""
        if self.prices.empty:
            raise ValueError(f'prices cannot be empty')
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError(f'prices index has to be of type pandas.DatetimeIndex')

    def _preprocessing(self,
                       asset_missing_threshold: float | None,
                       dates_missing_threshold: float | None) -> None:
        """
        Remove dates and assets with too many values missing
        :param asset_missing_threshold: remove Dates with more than asset_missing_threshold percent assets missing
        :param dates_missing_threshold: remove Assets with more than dates_missing_threshold percent dates missing

        1) Remove Assets (columns) with missing dates (nan) above dates_missing_threshold
        2) Remove Dates (rows) with missing assets (nan) above asset_missing_threshold
        """
        n, m = self.prices.shape

        # Remove assets with missing prices above threshold
        if dates_missing_threshold is not None:
            if not 0 < dates_missing_threshold < 1:
                raise ValueError(f'dates_missing_threshold has to be between 0 and 1')
            count_nan = self.prices.isna().sum(axis=0)
            to_drop = count_nan[count_nan > n * dates_missing_threshold].index
            if len(to_drop) > 0:
                self.prices.drop(to_drop, axis=1, inplace=True)
                self._info(f'{len(to_drop)} assets removed because they had more '
                           f'that {dates_missing_threshold * 100}% prices missing')

        # Remove dates with missing prices above threshold
        if asset_missing_threshold is not None:
            if not 0 < asset_missing_threshold < 1:
                raise ValueError(f'asset_missing_threshold has to be between 0 and 1')
            count_nan = self.prices.isna().sum(axis=1)
            to_drop = count_nan[count_nan > m * asset_missing_threshold].index
            if len(to_drop) > 0:
                self.prices.drop(to_drop, axis=0, inplace=True)
                self._info(f'{len(to_drop)} dates removed because they had more '
                           f'that {asset_missing_threshold * 100}% prices missing')

        # Forward fill missing values
        self.prices.fillna(method='ffill', inplace=True)
        # Backward fill missing values to the start
        self.prices.fillna(method='bfill', inplace=True)
        # Drop column if all its values are missing
        self.prices.dropna(axis=1, how='all', inplace=True)
        # Check that no nan are left
        if np.any(self.prices.isna()):
            raise ValueError(f'nan found in prices')

    def _remove_highly_correlated_assets(self,
                                         correlation_threshold: float | None) -> None:
        """
        When two assets have a correlation above correlation_threshold, we keep the asset with higher returns.
        Highly correlated assets increase calculus overhead and can cause matrix calculus errors without adding
        significant information
        :param correlation_threshold: correlation threshold
        """
        if correlation_threshold is None:
            return

        if not -1 <= correlation_threshold <= 1:
            raise ValueError(f'correlation_threshold has to be between -1 and 1')

        n = self.asset_nb
        to_remove = set()
        for i in range(n - 1):
            for j in range(i + 1, n):
                if self.corr[i, j] > correlation_threshold:
                    if i not in to_remove and j not in to_remove:
                        if self.mu[i] < self.mu[j]:
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
        self._info(f'{len(to_remove)} assets removed with a correlation above {correlation_threshold}')
        self.remove_assets(assets_to_remove=list(np.take(self.names, list(to_remove))))

    def remove_assets(self, assets_to_remove: list[str] | np.ndarray) -> None:
        self.prices.drop(assets_to_remove, axis=1, inplace=True)
        self.reset()

    def keep_assets(self, assets_to_keep: list[str] | np.ndarray) -> None:
        self.remove_assets(assets_to_remove=list(self.names[~np.isin(self.names, assets_to_keep)]))

    def reset(self):
        attrs = list(self.__dict__.keys())
        for attr in attrs:
            if attr[0] == '_':
                self.__setattr__(attr, None)
            elif attr not in ['verbose', 'name', 'start_date', 'end_date', 'prices']:
                self.__dict__.pop(attr, None)

    def set_expected_returns(self, expected_returns: np.ndarray) -> None:
        """
        Asset.expected_returns will return this custom expected return instead of the historical mean (Asset.mu)
        """
        if not isinstance(expected_returns, np.ndarray):
            raise ValueError(f'expected_returns has to be of type numpy.ndarray')
        if expected_returns.shape != (self.asset_nb,):
            raise ValueError(f'expected_returns must be of shape {(self.asset_nb,)}. '
                             f'But received {expected_returns.shape}')
        self._custom_expected_returns = expected_returns

    def set_expected_cov(self, expected_cov: np.ndarray) -> None:
        """
        Asset.expected_cov will return this custom expected covariance instead of the historical covariance (Asset.cov)
        """
        if not isinstance(expected_cov, np.ndarray):
            raise ValueError(f'expected_cov has to be of type numpy.ndarray')
        if expected_cov.shape != (self.asset_nb, self.asset_nb):
            raise ValueError(f'expected_returns must be of shape {(self.asset_nb, self.asset_nb)}. '
                             f'But received {expected_cov.shape}')
        self._custom_expected_cov = expected_cov

    @cached_property
    def returns(self) -> np.ndarray:
        """
        Compute simple returns from prices (R1 = S1/S0 - 1)
            --> simple returns leads to a better estimate of the efficient frontier than log returns
        And convert to numpy array of shape (assets, dates)
        """
        return self.prices.pct_change()[1:].to_numpy().T

    @cache
    def validate_returns(self) -> None:
        if not isinstance(self.returns, np.ndarray):
            raise TypeError(f'asset returns should be of type numpy.ndarray')
        if np.any(np.isnan(self.returns)):
            raise TypeError(f'returns should not contain nan')

    @cached_property
    def cumulative_returns(self) -> pd.DataFrame:
        """
        Compute the compounded cumulative returns (1+R1)*(1+R2)*(1+R3)...  = S1/S0 - 1)
        It's equivalent to rebasing prices to 1
        """
        df_returns = self.prices.pct_change()[1:]
        return (df_returns + 1).cumprod()

    @cached_property
    def mu(self) -> np.ndarray:
        return np.mean(self.returns, axis=1)

    @cached_property
    def std(self) -> np.ndarray:
        return np.std(self.returns, axis=1)

    @cached_property
    def cov(self) -> np.ndarray:
        return np.cov(self.returns)

    @property
    def expected_returns(self) -> np.ndarray:
        if self._custom_expected_returns is not None:
            return self._custom_expected_returns
        return self.mu

    @property
    def expected_cov(self) -> np.ndarray:
        if self._custom_expected_cov is not None:
            return self._custom_expected_cov
        return self.cov

    @cached_property
    def corr(self) -> np.ndarray:
        return np.corrcoef(self.returns)

    def dict_to_array(self, assets_dict: dict[str, float]) -> np.ndarray:
        """
        Take a dict of number per asset name and return a numpy array of values in the same order as the
        other assets arrays
        """
        missing_names = np.setdiff1d(list(assets_dict.keys()), self.names)
        if len(missing_names) > 0:
            logger.warning(f'The following names are not in the Assets universe {list(missing_names)}')

        return np.array([assets_dict.get(name, 0) for name in self.names])

    def plot_cumulative_returns(self,
                                names: list[str] | None = None,
                                idx: int | list[int] | slice = slice(None),
                                show: bool = True) -> go.Figure | None:
        if names is not None:
            fig = self.cumulative_returns[names].plot()
        else:
            fig = self.cumulative_returns.iloc[:, idx].plot()

        fig.update_layout(title='Compounded Cumulative Returns',
                          xaxis_title='Dates',
                          yaxis_title='Cumulative Returns',
                          legend_title_text='Assets')
        if show:
            fig.show()
        else:
            return fig

    def _info(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)
