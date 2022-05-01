import numpy as np
from enum import Enum
import uuid
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.utils.metrics import *

__all__ = ['FitnessType',
           'Metrics',
           'Portfolio']


class Metrics(Enum):
    MEAN = 'mean'
    STD = 'std'
    DOWNSIDE_STD = 'downside_std'
    ANNUALIZED_MEAN = 'annualized_mean'
    ANNUALIZED_STD = 'annualized_std'
    ANNUALIZED_DOWNSIDE_STD = 'annualized_downside_std'
    MAX_DRAWDOWN = 'max_drawdown'
    CDAR_95 = 'cdar_95'
    SHARPE_RATIO = 'sharpe_ratio'
    SORTINO_RATIO = 'sortino_ratio'
    CALMAR_RATIO = 'calmar_ratio'
    CDAR_95_RATIO = 'cdar_95_ratio'


class FitnessType(Enum):
    MEAN_STD = (Metrics.MEAN, Metrics.STD)
    MEAN_DOWNSIDE_STD = (Metrics.MEAN, Metrics.DOWNSIDE_STD)
    MEAN_DOWNSIDE_STD_MAX_DRAWDOWN = (Metrics.MEAN, Metrics.DOWNSIDE_STD, Metrics.MAX_DRAWDOWN)


class Portfolio:

    def __init__(self,
                 weights: np.ndarray,
                 fitness_type: FitnessType,
                 assets: Assets,
                 pid: str = None,
                 tag: str = 'ptf',
                 name: str = ''):

        # Sanity checks
        if assets.asset_nb != len(weights):
            raise ValueError(f'weights should be of size {assets.asset_nb}')
        if not isinstance(weights, np.ndarray):
            raise TypeError(f'weights should be of type numpy.ndarray')
        if abs(weights.sum() - 1) > 1e-5:
            raise TypeError(f'weights should sum to 1')

        if pid is None:
            self.pid = str(uuid.uuid4())
        else:
            self.pid = pid

        self.fitness_type = fitness_type
        self.weights = weights
        self.tag = tag
        self.name = name
        # Pointer to Assets
        self.assets = assets
        # Metrics
        self._returns = None
        self._prices_compounded = None
        self._prices_uncompounded = None
        self._mean = None
        self._std = None
        self._downside_std = None
        self._max_drawdown = None
        self._cdar_95 = None
        self._fitness = None

    @property
    def returns(self):
        if self._returns is None:
            self._returns = self.weights @ self.assets.returns
        return self._returns

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
            self._mean = self.weights @ self.assets.mu
        return self._mean

    @property
    def annualized_mean(self):
        return self.mean * AVG_TRADING_DAYS_PER_YEAR

    @property
    def std(self):
        if self._std is None:
            self._std = np.sqrt(self.weights @ self.assets.cov @ self.weights)
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

    def metrics(self):
        idx = [e.value for e in Metrics]
        res = [self.__getattribute__(attr) for attr in idx]
        return pd.DataFrame(res, index=idx, columns=['metrics'])

    def __str__(self):
        return f'Portfolio <{self.name} - {self.tag} - {self.pid} - {len(self.assets_names)} assets>'

    def __repr__(self):
        return str(self)
