import numpy as np
from enum import Enum

from portfolio_optimization.assets import *
from portfolio_optimization.utils.tools import *

__all__ = ['FitnessType',
           'Portfolio']


class FitnessType(Enum):
    MEAN_STD = ('mean', 'std')
    MEAN_DOWNSIDE_STD = ('mean', 'downside_st')
    MEAN_DOWNSIDE_STD_MAX_DRAWDOWN = ('mean', 'downside_std', 'max_drawdown')


class Portfolio:
    avg_trading_days_per_year = 255

    def __init__(self, weights: np.ndarray, fitness_type: FitnessType, assets: Assets, tag: str = 'ptf'):

        # Sanity checks
        if assets.asset_nb != len(weights):
            raise ValueError(f'weights should be of size {assets.asset_nb}')
        if not isinstance(weights, np.ndarray):
            raise TypeError(f'weights should be of type numpy.ndarray')
        if abs(weights.sum() - 1) > 1e-8:
            raise TypeError(f'weights should sum to 1')

        self.fitness_type = fitness_type
        self.weights = weights
        self.tag = tag
        # Pointer to Assets
        self.assets = assets
        # Metrics
        self._returns = None
        self._prices = None
        self._mu = None
        self._std = None
        self._downside_std = None
        self._max_drawdown = None
        self._fitness = None

    @property
    def returns(self):
        if self._returns is None:
            self._returns = self.weights @ self.assets.returns
        return self._returns

    @property
    def prices(self):
        if self._prices is None:
            self._prices = (self.returns + 1).cumprod()
        return self._prices

    @property
    def mu(self):
        if self._mu is None:
            self._mu = self.weights @ self.assets.mu
        return self._mu

    @property
    def annualized_mu(self):
        return self.mu * self.avg_trading_days_per_year

    @property
    def std(self):
        if self._std is None:
            self._std = np.sqrt(self.weights @ self.assets.cov @ self.weights)
        return self._std

    @property
    def annualized_std(self):
        return self.std * np.sqrt(self.avg_trading_days_per_year)

    @property
    def downside_std(self):
        if self._downside_std is None:
            self._downside_std = self.returns[self.returns < 0].std(ddof=1)
        return self._downside_std

    @property
    def annualized_downside_std(self):
        return self.downside_std * np.sqrt(self.avg_trading_days_per_year)

    @property
    def max_drawdown(self):
        if self._max_drawdown is None:
            self._max_drawdown = max_drawdown(self.prices)
        return self._max_drawdown

    @property
    def sharp_ratio(self):
        return self.annualized_mu / self.annualized_std

    @property
    def sortino_ratio(self):
        return self.annualized_mu / self.annualized_downside_std

    @property
    def fitness(self):
        """
        Fitness of the portfolio that contains the objectives to maximise and/or minimize .
=        """
        if self._fitness is None:
            if self.fitness_type == FitnessType.MEAN_STD:
                self._fitness = np.array([self.mu, -self.std])
            elif self.fitness_type == FitnessType.MEAN_DOWNSIDE_STD:
                self._fitness = np.array([self.mu, -self.downside_std])
            elif self.fitness_type == FitnessType.MEAN_DOWNSIDE_STD_MAX_DRAWDOWN:
                self._fitness = np.array([self.mu, -self.downside_std, -self.max_drawdown])
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
        return np.flatnonzero(self.weights != 0)

    @property
    def length(self):
        return np.count_nonzero(self.weights)

    def __str__(self):
        return f'Portfolio ({self.length} assets)'

    def __repr__(self):
        return str(self)
