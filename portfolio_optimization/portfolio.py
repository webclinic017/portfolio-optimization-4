import numpy as np
import uuid
import pandas as pd
from typing import Optional
import plotly.express as px
from functools import cached_property

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.utils.sorting import *
from portfolio_optimization.utils.metrics import *

__all__ = ['Portfolio',
           'MultiPeriodPortfolio']


class BasePortfolio:
    def __init__(self,
                 returns: np.array,
                 dates: np.array,
                 name: Optional[str] = None,
                 tag: Optional[str] = None,
                 fitness_type: FitnessType = FitnessType.MEAN_STD,
                 validate: bool = True):
        self.returns = returns
        self.dates = dates
        self.fitness_type = fitness_type
        if validate:
            self._validation()
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name
        if tag is None:
            self.tag = self.name
        else:
            self.tag = tag

    def _validation(self):
        if len(self.returns) != len(self.dates):
            raise ValueError(f'returns and dates should be of same size : {len(self.returns)} vs {len(self.dates)}')
        if not isinstance(self.returns, np.ndarray):
            raise TypeError('returns should be of type numpy.ndarray')
        if np.any(np.isnan(self.returns)):
            raise TypeError('returns should not contain nan')

    @cached_property
    def cumulative_returns(self):
        cumulative_returns = (self.returns + 1).cumprod()
        cumulative_returns = np.insert(cumulative_returns, 0, 1)
        return cumulative_returns

    @cached_property
    def cumulative_returns_uncompounded(self):
        returns = np.insert(self.returns, 0, 1)
        return np.cumsum(returns)

    @cached_property
    def returns_df(self):
        return pd.Series(index=self.dates, data=self.returns, name='returns')

    @cached_property
    def cumulative_returns_df(self):
        init_date = self.dates[0] - (self.dates[1] - self.dates[0])
        index = np.insert(self.dates, 0, init_date)
        return pd.Series(index=index, data=self.cumulative_returns, name='prices_compounded')

    @cached_property
    def cumulative_returns_uncompounded_df(self):
        init_date = self.dates[0] - (self.dates[1] - self.dates[0])
        index = np.insert(self.dates, 0, init_date)
        return pd.Series(index=index, data=self.cumulative_returns_uncompounded, name='prices_uncompounded')

    @cached_property
    def mean(self):
        return self.returns.mean()

    @property
    def annualized_mean(self):
        return self.mean * AVG_TRADING_DAYS_PER_YEAR

    @cached_property
    def std(self):
        return self.returns.std(ddof=1)

    @property
    def variance(self):
        return self.std ** 2

    @property
    def annualized_std(self):
        return self.std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @cached_property
    def downside_std(self):
        return downside_std(returns=self.returns)

    @property
    def downside_variance(self):
        return self.downside_std ** 2

    @property
    def annualized_downside_std(self):
        return self.downside_std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @cached_property
    def max_drawdown(self):
        return max_drawdown(prices=self.cumulative_returns)

    @cached_property
    def cdar_95(self):
        """
        Conditional Drawdown at Risk (CDaR) with a confidence level at 95%
        """
        return cdar(prices=self.cumulative_returns_uncompounded, beta=0.95)

    @cached_property
    def cvar_95(self):
        """
        Conditional historical Value at Risk (CVaR) with a confidence level at 95%
        """
        return cvar(returns=self.returns, beta=0.95)

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

    def summary(self, formatted: bool = True) -> pd.Series:
        summary_fmt = {
            'Mean (Expected Return)': (self.mean, '0.3%'),
            'Annualized Mean': (self.annualized_mean, '0.2%'),
            'Std (Volatility)': (self.std, '0.3%'),
            'Annualized Std': (self.annualized_std, '0.2%'),
            'Downside Std': (self.downside_std, '0.3%'),
            'Annualized Downside Std': (self.annualized_downside_std, '0.2%'),
            'Max Drawdown': (self.max_drawdown, '0.2%'),
            'CDaR at 95%': (self.cdar_95, '0.2%'),
            'CVaR at 95%': (self.cvar_95, '0.2%'),
            'Variance': (self.variance, '0.6%'),
            'Downside Variance': (self.downside_variance, '0.6%'),
            'Sharpe Ratio': (self.sharpe_ratio, '0.2f'),
            'Sortino Ratio': (self.sortino_ratio, '0.2f'),
            'Calmar Ratio': (self.calmar_ratio, '0.2f'),
            'Cdar at 95% Ratio': (self.cdar_95_ratio, '0.2f'),
            'Cvar at 95% Ratio': (self.cvar_95_ratio, '0.2f'),
        }

        if formatted:
            summary = {name: '{value:{fmt}}'.format(value=value, fmt=fmt) for name, (value, fmt) in summary_fmt.items()}
        else:
            summary = {name: value for name, (value, fmt) in summary_fmt.items()}

        return pd.Series(summary)

    @cached_property
    def fitness(self):
        """
        Fitness of the portfolio that contains the objectives to maximise and/or minimize .
=        """
        if self.fitness_type == FitnessType.MEAN_STD:
            return np.array([self.mean, -self.std])
        if self.fitness_type == FitnessType.MEAN_DOWNSIDE_STD:
            return np.array([self.mean, -self.downside_std])
        if self.fitness_type == FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN:
            return np.array([self.mean, -self.downside_std, -self.max_drawdown])
        raise ValueError(f'fitness_type {self.fitness_type} should be of type {FitnessType}')

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
        attrs = list(self.__dict__.keys())
        for attr in attrs:
            if attr[0] == '_':
                self.__setattr__(attr, None)
            elif attr not in ['returns', 'dates', 'name', 'tag', 'validate', 'fitness_type', 'weights', 'assets',
                              'portfolios']:
                self.__dict__.pop(attr, None)

    def reset_fitness(self, fitness_type: FitnessType):
        self.__dict__.pop('fitness', None)
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
        raise NotImplementedError

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
                 tag: Optional[str] = None,
                 fitness_type: FitnessType = FitnessType.MEAN_STD):

        self.assets = assets
        self.weights = weights
        returns = self.weights @ self.assets.returns
        self._validation()

        super().__init__(returns=returns,
                         dates=assets.dates[1:],
                         name=name,
                         tag=tag,
                         fitness_type=fitness_type,
                         validate=False)

    def _validation(self):
        self.assets.validate_returns()
        if not isinstance(self.weights, np.ndarray):
            raise TypeError(f'weights should be of type numpy.ndarray')
        if np.any(np.isnan(self.weights)):
            raise TypeError(f'weights should not contain nan')
        if self.assets.asset_nb != len(self.weights):
            raise ValueError(f'weights should be of size {self.assets.asset_nb}')

    @cached_property
    def mean(self):
        return self.weights @ self.assets.mu

    @cached_property
    def std(self):
        return np.sqrt(self.weights @ self.assets.cov @ self.weights)

    @property
    def sric(self):
        """
        Sharpe Ratio Information Criterion (SRIC) is an unbiased estimator of the sharpe ratio adjusting for both
        sources of bias which are noise fit and estimation error.
        Ref: Noise Fit, Estimation Error and a Sharpe Information Criterion. Dirk Paulsen (2019)
        """
        return self.sharpe_ratio - self.assets.asset_nb / (self.assets.date_nb * self.sharpe_ratio)

    @cached_property
    def assets_index(self):
        return np.flatnonzero(abs(self.weights) > ZERO_THRESHOLD)

    @cached_property
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

    @cached_property
    def length(self):
        return np.count_nonzero(abs(self.weights) > ZERO_THRESHOLD)

    def summary(self, formatted: bool = True) -> pd.Series:
        df = super().summary(formatted=formatted)
        assets_number = self.length
        if formatted:
            assets_number = str(int(assets_number))
        df['assets number'] = assets_number
        return df

    def get_weight(self, asset_name: str):
        try:
            return self.weights[np.where(self.assets.names == asset_name)[0][0]]
        except IndexError:
            raise IndexError(f'{asset_name} is not in the assets universe')

    def __str__(self):
        return f'Portfolio < {self.name} >'


class MultiPeriodPortfolio(BasePortfolio):

    def __init__(self,
                 portfolios: Optional[list[Portfolio]] = None,
                 name: Optional[str] = None,
                 tag: Optional[str] = None,
                 fitness_type: FitnessType = FitnessType.MEAN_STD):
        super().__init__(returns=np.array([]),
                         dates=np.array([]),
                         name=name,
                         tag=tag,
                         fitness_type=fitness_type,
                         validate=False)

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

    def summary(self, formatted: bool = True) -> pd.Series:
        df = super().summary(formatted=formatted)
        portfolios_number = len(self.portfolios)
        avg_assets_per_portfolio = np.mean(self.length)
        if formatted:
            portfolios_number = str(int(portfolios_number))
            avg_assets_per_portfolio = f'{avg_assets_per_portfolio:0.1f}'

        df['portfolios number'] = portfolios_number
        df['avg nb of assets per portfolio'] = avg_assets_per_portfolio
        return df

    def __str__(self):
        return f'MultiPeriodPortfolio < {self.name} >'

    def __repr__(self):
        return str(self)
