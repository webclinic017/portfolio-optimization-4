from collections.abc import Iterator
import numpy as np
import pandas as pd
import plotly.express as px
from functools import cache, cached_property
import numbers
import plotly.graph_objects as go
from portfolio_optimization.meta import Metrics, AVG_TRADING_DAYS_PER_YEAR, ZERO_THRESHOLD
from portfolio_optimization.assets import Assets
from portfolio_optimization.utils.sorting import dominate
from portfolio_optimization.utils.metrics import downside_std, max_drawdown, cvar, cdar

__all__ = ['BasePortfolio',
           'Portfolio',
           'MultiPeriodPortfolio']


class BasePortfolio:
    def __init__(self,
                 returns: np.ndarray,
                 dates: np.ndarray,
                 name: str | None = None,
                 tag: str | None = None,
                 fitness_metrics: list[Metrics] | None = None,
                 validate: bool = True,
                 cvar_beta: float = 0.95,
                 cdar_beta: float = 0.95):
        r"""
        Base Portfolio

        Parameters
        ----------
        returns: ndarray
                 The returns array of the Portfolio

        dates: ndarray
               The dates array of the portfolio. Must be of same size as the returns array.

        name: str | None, default None
              The name of the Portfolio. If None, the object id will be assigned to the name.
              When the Portfolio is added to a `Population`, the name will be frozen to avoid corrupting the
              `Population` hashmap.

        tag: str | None, default None
             A tag that can be used to manipulate group of Portfolios from a `Population`.

        fitness_metrics: list[Metrics] | None
                         A list of Fitness metrics used compute portfolio domination.
                         It is used the comparison of Portfolios and compute the `Population` pareto front.

        validate: bool, default True
                  If True, the Class attributes are validated
        """

        self._name = name if name is not None else str(id(self))
        self._frozen = False
        self._returns = returns
        self._dates = dates
        self.fitness_metrics = fitness_metrics if fitness_metrics is not None else [Metrics.MEAN, Metrics.STD]
        self.tag = tag if tag is not None else self._name
        self._cvar_beta = cvar_beta
        self._cdar_beta = cdar_beta

        if validate:
            self._validation()

    def __len__(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"<{type(self).__name__}, name: '{self.name}', tag: '{self.tag}'>"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}'>"

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.fitness_metrics), tuple(self.dates)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, BasePortfolio)
                and np.array_equal(self.fitness, other.fitness))

    def __gt__(self, other) -> bool:
        if not isinstance(other, BasePortfolio):
            raise TypeError(f">' not supported between instances of 'Portfolio' and '{type(other)}'")
        return self.dominates(other)

    def __ge__(self, other) -> bool:
        if not isinstance(other, BasePortfolio):
            raise TypeError(f">=' not supported between instances of 'Portfolio' and '{type(other)}'")
        return self.__eq__(other) or self.__gt__(other)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.unfreeze()
        return result

    def _validation(self) -> None:
        if len(self.returns) != len(self.dates):
            raise ValueError(f'returns and dates should be of same size : {len(self.returns)} vs {len(self.dates)}')
        if not isinstance(self.returns, np.ndarray):
            raise TypeError('returns should be of type numpy.ndarray')
        if np.any(np.isnan(self.returns)):
            raise TypeError('returns should not contain nan')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._frozen:
            raise AttributeError(f"can't set attribute 'name' after the Portfolio has been frozen (Portfolios are "
                                 f"frozen when they are added to a Population)")
        self._name = value

    @property
    def cvar_beta(self):
        return self._cvar_beta

    @cvar_beta.setter
    def cvar_beta(self, value: str) -> None:
        for attr in ['cvar', 'cvar_ratio']:
            self.__dict__.pop(attr, None)
        self._cvar_beta = value

    @property
    def cdar_beta(self):
        return self._cdar_beta

    @cdar_beta.setter
    def cdar_beta(self, value: str) -> None:
        for attr in ['cdar', 'cdar_ratio']:
            self.__dict__.pop(attr, None)
        self._cdar_beta = value

    @property
    def returns(self):
        return self._returns

    @property
    def dates(self):
        return self._dates

    @property
    def fitness_metrics(self) -> list[Metrics]:
        return self._fitness_metrics

    @fitness_metrics.setter
    def fitness_metrics(self, value: list[Metrics]) -> None:
        if not isinstance(value, list) or len(value) == 0:
            raise TypeError(f'fitness_metrics should be a non-empty list of Metrics')
        self._fitness_metrics = [Metrics(v) for v in value]
        self.__dict__.pop('fitness', None)

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    @cached_property
    def cumulative_returns(self) -> np.ndarray:
        cumulative_returns = (self.returns + 1).cumprod()
        cumulative_returns = np.insert(cumulative_returns, 0, 1)
        return cumulative_returns

    @cached_property
    def cumulative_returns_uncompounded(self) -> np.ndarray:
        returns = np.insert(self.returns, 0, 1)
        return np.cumsum(returns)

    @cached_property
    def returns_df(self) -> pd.Series:
        return pd.Series(index=self.dates, data=self.returns, name='returns')

    @cached_property
    def cumulative_returns_df(self) -> pd.Series:
        init_date = self.dates[0] - (self.dates[1] - self.dates[0])
        index = np.insert(self.dates, 0, init_date)
        return pd.Series(index=index, data=self.cumulative_returns, name='prices_compounded')

    @cached_property
    def cumulative_returns_uncompounded_df(self) -> pd.Series:
        init_date = self.dates[0] - (self.dates[1] - self.dates[0])
        index = np.insert(self.dates, 0, init_date)
        return pd.Series(index=index, data=self.cumulative_returns_uncompounded, name='prices_uncompounded')

    @cached_property
    def mean(self) -> float:
        return self.returns.mean()

    @property
    def annualized_mean(self) -> float:
        return self.mean * AVG_TRADING_DAYS_PER_YEAR

    # risks measures
    @cached_property
    def std(self) -> float:
        return self.returns.std(ddof=1)

    @property
    def annualized_std(self) -> float:
        return self.std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @property
    def variance(self) -> float:
        return self.std ** 2

    @property
    def annualized_variance(self) -> float:
        return self.annualized_std ** 2

    @cached_property
    def semi_std(self) -> float:
        return downside_std(returns=self.returns)

    @property
    def annualized_semi_std(self) -> float:
        return self.semi_std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @property
    def semi_variance(self) -> float:
        return self.semi_std ** 2

    @property
    def annualized_semi_variance(self) -> float:
        return self.annualized_semi_std ** 2

    @cached_property
    def max_drawdown(self) -> float:
        return max_drawdown(prices=self.cumulative_returns)

    @cached_property
    def cdar(self) -> float:
        """
        Conditional Drawdown at Risk (CDaR) with a confidence level at self.cdar_beta (default 95%)
        """
        return cdar(prices=self.cumulative_returns_uncompounded, beta=self.cdar_beta)

    @cached_property
    def cvar(self) -> float:
        """
        Conditional historical Value at Risk (CVaR) with a confidence level at self.cvar_beta (default 95%)
        """
        return cvar(returns=self.returns, beta=self.cvar_beta)

    @property
    def sharpe_ratio(self) -> float:
        return self.annualized_mean / self.annualized_std

    @property
    def sortino_ratio(self) -> float:
        return self.annualized_mean / self.annualized_semi_std

    @property
    def calmar_ratio(self) -> float:
        return self.annualized_mean / self.max_drawdown

    @property
    def cdar_ratio(self) -> float:
        return self.annualized_mean / self.cdar

    @property
    def cvar_ratio(self) -> float:
        return self.annualized_mean / self.cvar

    @property
    def composition(self) -> pd.DataFrame:
        raise NotImplementedError

    @cached_property
    def fitness(self) -> np.ndarray:
        """
        Fitness of the portfolio that contains the objectives to maximise and/or minimize .
        """
        res = []
        for metric in self.fitness_metrics:
            if metric in [Metrics.MEAN,
                          Metrics.ANNUALIZED_MEAN,
                          Metrics.SHARPE_RATIO,
                          Metrics.SORTINO_RATIO,
                          Metrics.CALMAR_RATIO,
                          Metrics.CDAR_RATIO,
                          Metrics.CVAR_RATIO]:
                sign = 1
            else:
                sign = -1
            res.append(sign * getattr(self, metric.value))
        return np.array(res)

    @cached_property
    def assets_index(self) -> np.ndarray:
        raise NotImplementedError

    def dominates(self, other, obj=slice(None)) -> bool:
        """
        Return true if each objective of the current portfolio's fitness is not strictly worse than
        the corresponding objective of the other portfolio's fitness and at least one objective is
        strictly better
        :param other: Other portfolio
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objective.
        """
        return dominate(self.fitness[obj], other.fitness[obj])

    def reset(self) -> None:
        attrs = list(self.__dict__.keys())
        for attr in attrs:
            if attr[0] != '_' and attr not in ['tag', 'validate']:
                self.__dict__.pop(attr, None)

    def metrics(self) -> pd.DataFrame:
        idx = [e.value for e in Metrics]
        res = [self.__getattribute__(attr) for attr in idx]
        return pd.DataFrame(res, index=idx, columns=['metrics'])

    def plot_cumulative_returns(self,
                                idx: int | slice = slice(None),
                                show: bool = True) -> go.Figure | None:
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
                                             idx: int | slice = slice(None),
                                             show: bool = True) -> go.Figure | None:
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
                     idx: int | slice = slice(None),
                     show: bool = True) -> go.Figure | None:
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
                            show: bool = True) -> go.Figure | None:
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

    def summary(self, formatted: bool = True) -> pd.Series:
        summary_fmt = {
            'Mean (Expected Return)': (self.mean, '0.3%'),
            'Annualized Mean': (self.annualized_mean, '0.2%'),
            'Std (Volatility)': (self.std, '0.3%'),
            'Annualized Std': (self.annualized_std, '0.2%'),
            'Downside Std': (self.semi_std, '0.3%'),
            'Annualized Downside Std': (self.annualized_semi_std, '0.2%'),
            'Max Drawdown': (self.max_drawdown, '0.2%'),
            'CDaR at 95%': (self.cdar, '0.2%'),
            'CVaR at 95%': (self.cvar, '0.2%'),
            'Variance': (self.variance, '0.6%'),
            'Downside Variance': (self.semi_variance, '0.6%'),
            'Sharpe Ratio': (self.sharpe_ratio, '0.2f'),
            'Sortino Ratio': (self.sortino_ratio, '0.2f'),
            'Calmar Ratio': (self.calmar_ratio, '0.2f'),
            'Cdar at 95% Ratio': (self.cdar_ratio, '0.2f'),
            'Cvar at 95% Ratio': (self.cvar_ratio, '0.2f'),
        }

        if formatted:
            summary = {name: '{value:{fmt}}'.format(value=value, fmt=fmt) for name, (value, fmt) in summary_fmt.items()}
        else:
            summary = {name: value for name, (value, fmt) in summary_fmt.items()}

        return pd.Series(summary)


class Portfolio(BasePortfolio):
    def __init__(self,
                 assets: Assets,
                 weights: np.ndarray,
                 previous_weights: np.ndarray | None = None,
                 transaction_costs: float | np.ndarray = 0,
                 name: str | None = None,
                 tag: str | None = None,
                 fitness_metrics: list[Metrics] | None = None):
        r"""
        Portfolio

        Parameters
        ----------
        assets: Assets
                The Assets object containing the assets market data and mean/covariance models

        weights: ndarray
                 The weights of the portfolio. They need to be of same size and same order as the Assets.

        previous_weights: ndarray | None, default None
                          The previous weights of the portfolio. They need to be of same size and same order as the
                          Assets. If transaction_cost is 0, it will have no impact on the portfolio returns.

        transaction_costs: float | ndarray, default 0
                           Transaction costs are fixed costs charged when you buy or sell an asset.
                           They are used to compute the cost of rebalancing the portfolio which is:
                                * transaction_costs * |previous_weights - new_weights|
                           They need to be a float or an array of same size and same order as the
                           Assets. A float means that all assets have identical transaction costs.
                           The total transaction cost is averaged over the entire period and impacted
                           on the return series.

        name: str | None, default None
              The name of the `Portfolio`. If None, the object id will be assigned to the name.
              When the `Portfolio` is added to a `Population`, the name will be frozen to avoid corrupting the
              `Population` hashmap.

        tag: str | None, default None
             A tag that can be used to manipulate group of `Portfolios` from a `Population`.

        fitness_metrics: list[Metrics] | None
                         A list of Fitness metrics used compute portfolio domination.
                         It is used the comparison of `Portfolios` and compute the `Population` pareto front.

        """
        self._assets = assets
        self._weights = weights
        if previous_weights is None:
            self._previous_weights = np.zeros(self.assets.asset_nb)
        else:
            self._previous_weights = previous_weights
        self._transaction_costs = transaction_costs
        self._validation()

        portfolio_returns = self.weights @ self.assets.returns
        if np.isscalar(self.transaction_costs) and self.transaction_costs == 0:
            costs = 0
        else:
            costs = (self.transaction_costs * abs(self.previous_weights - self.weights)).sum()

        super().__init__(returns=portfolio_returns - costs / len(portfolio_returns),
                         dates=assets.dates[1:],
                         name=name,
                         tag=tag,
                         fitness_metrics=fitness_metrics,
                         validate=False)

    @cache
    def __len__(self) -> int:
        return np.count_nonzero(abs(self.weights) > ZERO_THRESHOLD)

    def __neg__(self):
        return Portfolio(weights=-self.weights, assets=self.assets)

    def __abs__(self):
        return Portfolio(weights=np.abs(self.weights), assets=self.assets)

    def __round__(self, n: int):
        return Portfolio(weights=np.round(self.weights, n), assets=self.assets)

    def __floor__(self):
        return Portfolio(weights=np.floor(self.weights), assets=self.assets)

    def __trunc__(self):
        return Portfolio(weights=np.trunc(self.weights), assets=self.assets)

    def __add__(self, other):
        if not isinstance(other, Portfolio):
            raise TypeError(f'Cannot add a Portfolio with an object of type {type(other)}')
        if self.assets != other.assets:
            raise ValueError(f'To add two Portfolios, there Assets should be the same object')
        return Portfolio(weights=self.weights + other.weights, assets=self.assets)

    def __sub__(self, other):
        if not isinstance(other, Portfolio):
            raise TypeError(f'Cannot subtract a Portfolio with an object of type {type(other)}')
        if self.assets != other.assets:
            raise ValueError(f'To subtract two Portfolios, there Assets should be the same object')
        return Portfolio(weights=self.weights - other.weights, assets=self.assets)

    def __mul__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(f'Portfolio can only be multiplied by a number, but received a {type(other)}')
        return Portfolio(weights=other * self.weights, assets=self.assets)

    __rmul__ = __mul__

    def __floordiv__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(f'Portfolio can only be floor divided by a number, but received a {type(other)}')
        return Portfolio(weights=np.floor_divide(self.weights, other), assets=self.assets)

    def __truediv__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(f'Portfolio can only be divided by a number, but received a {type(other)}')
        return Portfolio(weights=self.weights / other, assets=self.assets)

    def _validation(self) -> None:
        self.assets.validate_returns()
        if not isinstance(self.weights, np.ndarray):
            raise TypeError(f'weights should be of type numpy.ndarray')
        if np.any(np.isnan(self.weights)):
            raise TypeError(f'weights should not contain nan')
        if self.assets.asset_nb != len(self.weights):
            raise ValueError(f'weights should be of size {self.assets.asset_nb}')
        if not isinstance(self.previous_weights, np.ndarray):
            raise TypeError(f'previous_weights should be of type numpy.ndarray')
        if np.any(np.isnan(self.previous_weights)):
            raise TypeError(f'previous_weights should not contain nan')
        if self.assets.asset_nb != len(self.previous_weights):
            raise ValueError(f'previous_weights should be of size {self.assets.asset_nb}')
        if not np.isscalar(self.transaction_costs):
            if not isinstance(self.transaction_costs, np.ndarray):
                raise TypeError(f'transaction_costs should be of type numpy.ndarray or float')
            if np.any(np.isnan(self.transaction_costs)):
                raise TypeError(f'transaction_costs should not contain nan')
            if self.assets.asset_nb != len(self.transaction_costs):
                raise ValueError(f'transaction_costs should be of size {self.assets.asset_nb}')

    @property
    def assets(self) -> Assets:
        return self._assets

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def previous_weights(self) -> np.ndarray:
        return self._previous_weights

    @property
    def transaction_costs(self) -> float | np.ndarray:
        return self._transaction_costs

    @property
    def sric(self) -> float:
        r"""
        Sharpe Ratio Information Criterion (SRIC) is an unbiased estimator of the sharpe ratio adjusting for both
        sources of bias which are noise fit and estimation error.
        Ref: Noise Fit, Estimation Error and a Sharpe Information Criterion. Dirk Paulsen (2019)
        """
        return self.sharpe_ratio - self.assets.asset_nb / (self.assets.date_nb * self.sharpe_ratio)

    @cached_property
    def assets_index(self) -> np.ndarray:
        return np.flatnonzero(abs(self.weights) > ZERO_THRESHOLD)

    @cached_property
    def assets_names(self) -> np.ndarray:
        return self.assets.names[self.assets_index]

    @property
    def composition(self) -> pd.DataFrame:
        weights = self.weights[self.assets_index]
        df = pd.DataFrame({'asset': self.assets_names, 'weight': weights})
        df.sort_values(by='weight', ascending=False, inplace=True)
        df.rename(columns={'weight': self.name}, inplace=True)
        df.set_index('asset', inplace=True)
        return df

    def reset(self) -> None:
        super().reset()
        self.__len__.cache_clear()

    def summary(self, formatted: bool = True) -> pd.Series:
        df = super().summary(formatted=formatted)
        assets_number = len(self)
        if formatted:
            assets_number = str(int(assets_number))
        df['assets number'] = assets_number
        return df

    def get_weight(self, asset_name: str) -> float:
        try:
            return self.weights[np.where(self.assets.names == asset_name)[0][0]]
        except IndexError:
            raise IndexError(f'{asset_name} is not in the assets universe')


class MultiPeriodPortfolio(BasePortfolio):
    def __init__(self,
                 portfolios: list[Portfolio] | None = None,
                 name: str | None = None,
                 tag: str | None = None,
                 fitness_metrics: list[Metrics] | None = None):
        r"""
        Multi Period Portfolio

        Parameters
        ----------

        portfolios: list[Portfolio] | None, default None
                    A list of Portfolios. If None, MultiPeriodPortfolio will be initialized with an empty list.

        name: str | None, default None
              The name of the `Portfolio`. If None, the object id will be assigned to the name.
              When the `Portfolio` is added to a `Population`, the name will be frozen to avoid corrupting the
              `Population` hashmap.

        tag: str | None, default None
             A tag that can be used to manipulate group of `Portfolios` from a `Population`.

         fitness_metrics: list[Metrics] | None
                          A list of Fitness metrics used compute portfolio domination.
                          It is used the comparison of `Portfolios` and compute the `Population` pareto front.
        """

        portfolios, returns, dates = self._initialize(portfolios=portfolios)
        self._portfolios = portfolios
        super().__init__(returns=returns,
                         dates=dates,
                         name=name,
                         tag=tag,
                         fitness_metrics=fitness_metrics,
                         validate=False)

    @staticmethod
    def _initialize(portfolios: list[Portfolio] | None = None) -> tuple[list[Portfolio], np.ndarray, np.ndarray]:
        if portfolios is not None and len(portfolios) != 0:
            iteration = iter(portfolios)
            prev_p = next(iteration)
            while (p := next(iteration, None)) is not None:
                if p.dates[0] <= prev_p.dates[-1]:
                    raise ValueError(f'Portfolios dates should not overlap: {p} overlapping {prev_p}')
                prev_p = p
            returns = np.concatenate([p.returns for p in portfolios], axis=0)
            dates = np.concatenate([p.dates for p in portfolios], axis=0)
        else:
            returns = np.array([])
            dates = np.array([])
            portfolios = []
        return portfolios, returns, dates

    @property
    def portfolios(self) -> list[Portfolio]:
        return self._portfolios

    @portfolios.setter
    def portfolios(self, value: list[Portfolio] | None = None):
        portfolios, returns, dates = self._initialize(portfolios=value)
        self._portfolios = portfolios
        self._returns = returns
        self._dates = dates
        self.reset()

    def __len__(self) -> int:
        return len(self._portfolios)

    def __getitem__(self, key: int | slice) -> Portfolio | list[Portfolio]:
        return self._portfolios[key]

    def __setitem__(self, key: int, value: Portfolio) -> None:
        if not isinstance(value, Portfolio):
            raise TypeError(f'Cannot set a value with type {type(value)}')
        new_portfolios = self._portfolios.copy()
        new_portfolios[key] = value
        portfolios, returns, dates = self._initialize(portfolios=new_portfolios)
        self._portfolios = portfolios
        self._returns = returns
        self._dates = dates
        self.reset()

    def __delitem__(self, key: int) -> None:
        new_portfolios = self._portfolios.copy()
        del new_portfolios[key]
        portfolios, returns, dates = self._initialize(portfolios=new_portfolios)
        self._portfolios = portfolios
        self._returns = returns
        self._dates = dates
        self.reset()

    def __iter__(self) -> Iterator[Portfolio]:
        return iter(self._portfolios)

    def __contains__(self, value: Portfolio) -> bool:
        if not isinstance(value, Portfolio):
            return False
        return value in self._portfolios

    def __neg__(self):
        return MultiPeriodPortfolio(portfolios=[-p for p in self], tag=self.tag, fitness_metrics=self.fitness_metrics)

    def __abs__(self):
        return MultiPeriodPortfolio(portfolios=[abs(p) for p in self], tag=self.tag,
                                    fitness_metrics=self.fitness_metrics)

    def __round__(self, n: int):
        return MultiPeriodPortfolio(portfolios=[p.__round__(n) for p in self], tag=self.tag,
                                    fitness_metrics=self.fitness_metrics)

    def __floor__(self):
        return MultiPeriodPortfolio(portfolios=[np.floor(p) for p in self], tag=self.tag,
                                    fitness_metrics=self.fitness_metrics)

    def __trunc__(self):
        return MultiPeriodPortfolio(portfolios=[np.trunc(p) for p in self], tag=self.tag,
                                    fitness_metrics=self.fitness_metrics)

    def __add__(self, other):
        if not isinstance(other, MultiPeriodPortfolio):
            raise TypeError(f'Cannot add a MultiPeriodPortfolio with an object of type {type(other)}')
        if len(self) != len(other):
            raise TypeError(f'Cannot add two MultiPeriodPortfolio of different sizes')
        return MultiPeriodPortfolio(portfolios=[p1 + p2 for p1, p2 in zip(self, other)], tag=self.tag,
                                    fitness_metrics=self.fitness_metrics)

    def __sub__(self, other):
        if not isinstance(other, MultiPeriodPortfolio):
            raise TypeError(f'Cannot subtract a MultiPeriodPortfolio with an object of type {type(other)}')
        if len(self) != len(other):
            raise TypeError(f'Cannot subtract two MultiPeriodPortfolio of different sizes')
        return MultiPeriodPortfolio(portfolios=[p1 - p2 for p1, p2 in zip(self, other)], tag=self.tag,
                                    fitness_metrics=self.fitness_metrics)

    def __mul__(self, other: numbers.Number | list[numbers.Number] | np.ndarray):
        if np.isscalar(other):
            portfolios = [p * other for p in self]
        else:
            portfolios = [p * a for p, a in zip(self, other)]
        return MultiPeriodPortfolio(portfolios=portfolios, tag=self.tag, fitness_metrics=self.fitness_metrics)

    __rmul__ = __mul__

    def __floordiv__(self, other: numbers.Number | list[numbers.Number] | np.ndarray):
        if np.isscalar(other):
            portfolios = [p // other for p in self]
        else:
            portfolios = [p // a for p, a in zip(self, other)]
        return MultiPeriodPortfolio(portfolios=portfolios, tag=self.tag, fitness_metrics=self.fitness_metrics)

    def __truediv__(self, other: numbers.Number | list[numbers.Number] | np.ndarray):
        if np.isscalar(other):
            portfolios = [p / other for p in self]
        else:
            portfolios = [p / a for p, a in zip(self, other)]
        return MultiPeriodPortfolio(portfolios=portfolios, tag=self.tag, fitness_metrics=self.fitness_metrics)

    def append(self, portfolio: Portfolio) -> None:
        if len(self) != 0:
            start_date = portfolio.dates[0]
            prev_last_date = self[-1].dates[-1]
            if start_date < prev_last_date:
                raise ValueError(f'Portfolios dates should not overlap: {prev_last_date} -> {start_date} ')
        self._portfolios.append(portfolio)
        self._returns = np.concatenate([self.returns, portfolio.returns], axis=0)
        self._dates = np.concatenate([self.dates, portfolio.dates], axis=0)
        self.reset()

    @property
    def assets_index(self) -> np.ndarray:
        return np.array([p.assets_index for p in self])

    @property
    def assets_names(self) -> np.ndarray:
        return np.array([p.assets_names for p in self])

    @property
    def composition(self) -> pd.DataFrame:
        df = pd.concat([p.composition for p in self], axis=1)
        df.fillna(0, inplace=True)
        return df

    def summary(self, formatted: bool = True) -> pd.Series:
        df = super().summary(formatted=formatted)
        portfolios_number = len(self)
        avg_assets_per_portfolio = np.mean([len(p) for p in self])
        if formatted:
            portfolios_number = str(int(portfolios_number))
            avg_assets_per_portfolio = f'{avg_assets_per_portfolio:0.1f}'
        df['portfolios number'] = portfolios_number
        df['avg nb of assets per portfolio'] = avg_assets_per_portfolio
        return df
