import numpy as np
import uuid
import pandas as pd
from typing import Optional
import plotly.express as px

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.utils.metrics import *

__all__ = ['Portfolio',
           'MultiPeriodPortfolio']


class BasePortfolio:

    def __init__(self,
                 returns: np.array,
                 dates: np.array,
                 name: Optional[str] = None,
                 tag: str = 'portfolio',
                 fitness_type: FitnessType = FitnessType.MEAN_STD):

        self.returns = returns
        self.dates = dates
        self.fitness_type = fitness_type
        self._validation()

        # Ids
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name
        self.tag = tag

        # Prices
        self._cumulative_returns = None
        self._cumulative_returns_uncompounded = None

        # Metrics
        self._mean = None
        self._std = None
        self._downside_std = None
        self._max_drawdown = None
        self._cdar_95 = None
        self._cvar_95 = None
        self._fitness = None

    def _validation(self):
        if len(self.returns) != len(self.dates):
            raise ValueError(f'returns and dates should be of same size : {len(self.returns)} vs {len(self.dates)}')
        if not isinstance(self.returns, np.ndarray):
            raise TypeError('returns should be of type numpy.ndarray')
        if np.any(np.isnan(self.returns)):
            raise TypeError('returns should not contain nan')

    @property
    def cumulative_returns(self):
        if self._cumulative_returns is None:
            cumulative_returns = (self.returns + 1).cumprod()
            self._cumulative_returns = np.insert(cumulative_returns, 0, 1)
        return self._cumulative_returns

    @property
    def cumulative_returns_uncompounded(self):
        if self._cumulative_returns_uncompounded is None:
            returns = np.insert(self.returns, 0, 1)
            self._cumulative_returns_uncompounded = np.cumsum(returns)
        return self._cumulative_returns_uncompounded

    @property
    def returns_df(self):
        return pd.Series(index=self.dates, data=self.returns, name='returns')

    @property
    def cumulative_returns_df(self):
        init_date = self.dates[0] - (self.dates[1] - self.dates[0])
        index = np.insert(self.dates, 0, init_date)
        return pd.Series(index=index, data=self.cumulative_returns, name='prices_compounded')

    @property
    def cumulative_returns_uncompounded_df(self):
        init_date = self.dates[0] - (self.dates[1] - self.dates[0])
        index = np.insert(self.dates, 0, init_date)
        return pd.Series(index=index, data=self.cumulative_returns_uncompounded, name='prices_uncompounded')

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self.returns.mean()
        return self._mean

    @property
    def annualized_mean(self):
        return self.mean * AVG_TRADING_DAYS_PER_YEAR

    @property
    def std(self):
        if self._std is None:
            self._std = self.returns.std(ddof=1)
        return self._std

    @property
    def variance(self):
        return self.std ** 2

    @property
    def annualized_std(self):
        return self.std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @property
    def downside_std(self):
        if self._downside_std is None:
            self._downside_std = downside_std(returns=self.returns)
        return self._downside_std

    @property
    def downside_variance(self):
        return self.downside_std ** 2

    @property
    def annualized_downside_std(self):
        return self.downside_std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @property
    def max_drawdown(self):
        if self._max_drawdown is None:
            self._max_drawdown = max_drawdown(prices=self.cumulative_returns)
        return self._max_drawdown

    @property
    def cdar_95(self):
        """
        Conditional Drawdown at Risk (CDaR) with a confidence level at 95%
        """
        if self._cdar_95 is None:
            self._cdar_95 = cdar(prices=self.cumulative_returns_uncompounded, beta=0.95)
        return self._cdar_95

    @property
    def cvar_95(self):
        """
        Conditional historical Value at Risk (CVaR) with a confidence level at 95%
        """
        if self._cvar_95 is None:
            self._cvar_95 = cvar(returns=self.returns, beta=0.95)
        return self._cvar_95

    @property
    def sharpe_ratio(self):
        return self.annualized_mean / self.annualized_std

    @property
    def sortino_ratio(self):
        return self.annualized_mean / self.annualized_downside_std

    @property
    def calmar_ratio(self):
        return self.annualized_mean / self.max_drawdown

    @property
    def cdar_95_ratio(self):
        return self.annualized_mean / self.cdar_95

    @property
    def cvar_95_ratio(self):
        return self.annualized_mean / self.cvar_95

    @property
    def fitness(self):
        """
        Fitness of the portfolio that contains the objectives to maximise and/or minimize .
=        """
        if self._fitness is None:
            if self.fitness_type == FitnessType.MEAN_STD:
                self._fitness = np.array([self.mean, -self.std])
            elif self.fitness_type == FitnessType.MEAN_DOWNSIDE_STD:
                self._fitness = np.array([self.mean, -self.downside_std])
            elif self.fitness_type == FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN:
                self._fitness = np.array([self.mean, -self.downside_std, -self.max_drawdown])
            else:
                raise ValueError(f'fitness_type {self.fitness_type} should be of type {FitnessType}')
        return self._fitness

    def dominates(self, other, obj=slice(None)):
        """
        Return true if each objective of the current portfolio's fitness is not strictly worse than
        the corresponding objective of the other portfolio's fitness and at least one objective is
        strictly better.
        :param other: Other portfolio
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        return dominate(self.fitness[obj], other.fitness[obj])

    def reset_metrics(self):
        for attr in self.__dict__.keys():
            if attr[0] == '_':
                self.__setattr__(attr, None)

    def reset_fitness(self, fitness_type: FitnessType):
        self._fitness = None
        self.fitness_type = fitness_type

    def metrics(self):
        idx = [e.value for e in Metrics]
        res = [self.__getattribute__(attr) for attr in idx]
        return pd.DataFrame(res, index=idx, columns=['metrics'])

    def plot_cumulative_returns(self,
                                idx=slice(None),
                                show: bool = True):
        fig = self.cumulative_returns_df.iloc[idx].plot()
        fig.update_layout(title='Prices Compounded',
                          xaxis_title='Dates',
                          yaxis_title='Prices',
                          showlegend=False)
        if show:
            fig.show()
        else:
            return fig

    def plot_cumulative_returns_uncompounded(self,
                                             idx=slice(None),
                                             show: bool = True):
        fig = self.cumulative_returns_uncompounded_df.iloc[idx].plot()
        fig.update_layout(title='Prices Uncompounded',
                          xaxis_title='Dates',
                          yaxis_title='Prices',
                          showlegend=False)
        if show:
            fig.show()
        else:
            return fig

    def plot_returns(self,
                     idx=slice(None),
                     show: bool = True):
        fig = self.returns_df.iloc[idx].plot()
        fig.update_layout(title='Returns',
                          xaxis_title='Dates',
                          yaxis_title='Returns',
                          showlegend=False)
        if show:
            fig.show()
        else:
            return fig

    def plot_rolling_sharpe(self,
                            days: int = 30,
                            show: bool = True):
        s = pd.Series(self.returns, index=self.dates)
        rolling = s.rolling(window=days)
        rolling_sharpe = np.sqrt(AVG_TRADING_DAYS_PER_YEAR) * rolling.mean() / rolling.std(ddof=1)
        rolling_sharpe.name = f'Sharpe {days} days'
        fig = rolling_sharpe.plot()
        fig.add_hline(y=self.sharpe_ratio, line_width=1, line_dash='dash', line_color='blue')
        fig.add_hrect(y0=0, y1=rolling_sharpe.max() * 1.3, line_width=0, fillcolor='green', opacity=0.1)
        fig.add_hrect(y0=rolling_sharpe.min() * 1.3, y1=0, line_width=0, fillcolor='red', opacity=0.1)

        fig.update_layout(title=f'Rolling Sharpe - {days} days',
                          xaxis_title='Dates',
                          yaxis_title='Sharpe Ratio',
                          showlegend=False)
        if show:
            fig.show()
        else:
            return fig

    @property
    def composition(self) -> pd.DataFrame:
        """Implemented in Portfolio and MultiPeriodPortfolio"""
        return pd.DataFrame()

    def plot_composition(self, show: bool = True):
        df = self.composition.T
        fig = px.bar(df, x=df.index, y=df.columns)
        fig.update_layout(title='Portfolio Composition',
                          xaxis_title='Portfolio',
                          yaxis_title='Weight',
                          legend_title_text='Assets')
        if show:
            fig.show()
        else:
            return fig


class Portfolio(BasePortfolio):

    def __init__(self,
                 weights: np.ndarray,
                 assets: Assets,
                 name: Optional[str] = None,
                 tag: str = 'ptf',
                 fitness_type: FitnessType = FitnessType.MEAN_STD):

        self.assets = assets
        self.weights = weights
        returns = self.weights @ self.assets.returns
        self._validation()

        super().__init__(returns=returns,
                         dates=assets.dates[1:],
                         name=name,
                         tag=tag,
                         fitness_type=fitness_type)

    def _validation(self):
        for name, value in [('weights', self.weights),
                            ('assets.returns', self.assets.returns)]:
            if not isinstance(value, np.ndarray):
                raise TypeError(f'{name} should be of type numpy.ndarray')
            if np.any(np.isnan(value)):
                raise TypeError(f'{name} should not contain nan')

        if self.assets.asset_nb != len(self.weights):
            raise ValueError(f'weights should be of size {self.assets.asset_nb}')

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self.weights @ self.assets.mu
        return self._mean

    @property
    def std(self):
        if self._std is None:
            self._std = np.sqrt(self.weights @ self.assets.cov @ self.weights)
        return self._std

    @property
    def sric(self):
        """
        Sharpe Ratio Information Criterion (SRIC) is an unbiased estimator of the sharpe ratio adjusting for both
        sources of bias which are noise fit and estimation error.
        Ref: Noise Fit, Estimation Error and a Sharpe Information Criterion. Dirk Paulsen (2019)
        """
        return self.sharpe_ratio - self.assets.asset_nb / (self.assets.date_nb * self.sharpe_ratio)

    @property
    def assets_index(self):
        return np.flatnonzero(abs(self.weights) > ZERO_THRESHOLD)

    @property
    def assets_names(self):
        return self.assets.names[self.assets_index]

    @property
    def composition(self):
        weights = self.weights[self.assets_index]
        df = pd.DataFrame({'asset': self.assets_names, 'weight': weights})
        df.sort_values(by='weight', ascending=False, inplace=True)
        df.rename(columns={'weight': self.name}, inplace=True)
        df.set_index('asset', inplace=True)
        return df

    @property
    def length(self):
        return np.count_nonzero(abs(self.weights) > ZERO_THRESHOLD)

    def get_weight(self, asset_name: str):
        try:
            return self.weights[np.where(self.assets.names == asset_name)[0][0]]
        except IndexError:
            raise IndexError(f'{asset_name} is not in the assets universe')

    def __str__(self):
        return f'Portfolio < {self.name} >'

    def __repr__(self):
        return str(self)


class MultiPeriodPortfolio(BasePortfolio):

    def __init__(self,
                 portfolios: Optional[list[Portfolio]] = None,
                 name: Optional[str] = None,
                 tag: str = 'multi-period-portfolio',
                 fitness_type: FitnessType = FitnessType.MEAN_STD):
        super().__init__(returns=np.array([]),
                         dates=np.array([]),
                         name=name,
                         tag=tag,
                         fitness_type=fitness_type)

        # Ensure that Portfolios dates do not overlap
        self.portfolios = []
        if portfolios is not None:
            for portfolio in portfolios:
                self.add(portfolio)

    def add(self, portfolio: Portfolio):
        if self.portfolios is None:
            return
        if len(self.portfolios) != 0:
            start_date = portfolio.assets.dates[0]
            prev_last_date = self.portfolios[-1].assets.dates[-1]
            if start_date < prev_last_date:
                raise ValueError(f'Portfolios dates should not overlap: {prev_last_date} -> {start_date} ')
        self.portfolios.append(portfolio)
        self.returns = np.concatenate([self.returns, portfolio.returns], axis=0)
        self.dates = np.concatenate([self.dates, portfolio.dates], axis=0)
        self.reset_metrics()

    @property
    def assets_index(self):
        return np.array([portfolio.assets_index for portfolio in self.portfolios])

    @property
    def assets_names(self):
        return np.array([portfolio.assets_names for portfolio in self.portfolios])

    @property
    def composition(self):
        df = pd.concat([portfolio.composition for portfolio in self.portfolios], axis=1)
        df.fillna(0, inplace=True)
        return df

    @property
    def length(self):
        return [portfolio.length for portfolio in self.portfolios]

    def __str__(self):
        return f'MultiPeriodPortfolio < {self.name} >'

    def __repr__(self):
        return str(self)
