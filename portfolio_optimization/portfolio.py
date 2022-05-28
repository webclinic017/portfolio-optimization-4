import numpy as np
import uuid
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.meta import Metrics, FitnessType
from portfolio_optimization.utils.tools import *
from portfolio_optimization.utils.metrics import *

__all__ = ['Portfolio',
           'MultiPeriodPortfolio']


class BasePortfolio:

    def __init__(self,
                 returns: np.array,
                 pid: str = None,
                 tag: str = 'portfolio',
                 name: str = '',
                 fitness_type: FitnessType = FitnessType.MEAN_STD):

        self.returns = returns
        self.fitness_type = fitness_type

        # Ids
        if pid is None:
            self.pid = str(uuid.uuid4())
        else:
            self.pid = pid
        self.tag = tag
        self.name = name

        # Prices
        self._prices_compounded = None
        self._prices_uncompounded = None

        # Metrics
        self._mean = None
        self._std = None
        self._downside_std = None
        self._max_drawdown = None
        self._cdar_95 = None
        self._cvar_95 = None
        self._fitness = None

    @property
    def prices_compounded(self):
        if self._prices_compounded is None:
            prices_compounded = (self.returns + 1).cumprod()
            self._prices_compounded = np.insert(prices_compounded, 0, 1)
        return self._prices_compounded

    @property
    def prices_uncompounded(self):
        if self._prices_uncompounded is None:
            returns = np.insert(self.returns, 0, 1)
            self._prices_uncompounded = np.cumsum(returns)
        return self._prices_uncompounded

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
    def annualized_std(self):
        return self.std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @property
    def downside_std(self):
        if self._downside_std is None:
            self._downside_std = downside_std(returns=self.returns)
        return self._downside_std

    @property
    def annualized_downside_std(self):
        return self.downside_std * np.sqrt(AVG_TRADING_DAYS_PER_YEAR)

    @property
    def max_drawdown(self):
        if self._max_drawdown is None:
            self._max_drawdown = max_drawdown(prices=self.prices_compounded)
        return self._max_drawdown

    @property
    def cdar_95(self):
        """
        Conditional Drawdown at Risk (CDaR) with a confidence level at 95%
        """
        if self._cdar_95 is None:
            self._cdar_95 = cdar(prices=self.prices_uncompounded, beta=0.95)
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


class Portfolio(BasePortfolio):

    def __init__(self,
                 weights: np.ndarray,
                 assets: Assets,
                 pid: str = None,
                 tag: str = 'ptf',
                 name: str = '',
                 fitness_type: FitnessType = FitnessType.MEAN_STD):

        # Sanity checks
        if assets.asset_nb != len(weights):
            raise ValueError(f'weights should be of size {assets.asset_nb}')
        if not isinstance(weights, np.ndarray):
            raise TypeError(f'weights should be of type numpy.ndarray')
        if abs(weights.sum() - 1) > 1e-5:
            raise TypeError(f'weights should sum to 1')

        self.assets = assets
        self.weights = weights
        returns = self.weights @ self.assets.returns

        super().__init__(returns=returns,
                         pid=pid,
                         tag=tag,
                         name=name,
                         fitness_type=fitness_type)

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
        df = pd.DataFrame({'name': self.assets_names, 'weight': weights})
        df.sort_values(by='weight', ascending=False, inplace=True)
        df.set_index('name', inplace=True)
        return df

    @property
    def length(self):
        return np.count_nonzero(abs(self.weights) > ZERO_THRESHOLD)

    def __str__(self):
        return f'Portfolio < {self.pid} - {self.name} - {self.tag} - {len(self.assets_names)} assets>'

    def __repr__(self):
        return str(self)


class MultiPeriodPortfolio(BasePortfolio):

    def __init__(self,
                 portfolios: list[Portfolio],
                 pid: str = None,
                 tag: str = 'multi-period-portfolio',
                 name: str = '',
                 fitness_type: FitnessType = FitnessType.MEAN_STD):
        # Ensure that Portfolios dates do not overlap
        self.portfolios = [portfolios[0]]
        for portfolio in portfolios[1:]:
            start_date = portfolio.assets.dates[0]
            prev_last_date = self.portfolios[-1].assets.dates[-1]
            if start_date < prev_last_date:
                raise ValueError(f'Portfolios dates should not overlap: {prev_last_date} -> {start_date} ')
            self.portfolios.append(portfolio)

        returns = np.concatenate([portfolio.returns for portfolio in self.portfolios], axis=0)

        super().__init__(returns=returns,
                         pid=pid,
                         tag=tag,
                         name=name,
                         fitness_type=fitness_type)

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
        df.columns = [f'weight_{portfolio.assets.start_date}_{portfolio.assets.end_date}'
                      for portfolio in self.portfolios]
        return df

    @property
    def length(self):
        return [portfolio.length for portfolio in self.portfolios]

    def __str__(self):
        return f'MultiPeriodPortfolio <{self.pid} - {self.name} - {self.tag}>'

    def __repr__(self):
        return str(self)
