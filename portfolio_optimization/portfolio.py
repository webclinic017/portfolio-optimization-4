import numpy as np
import pandas as pd
from enum import Enum

from portfolio_optimization.utils.preprocessing import *
from portfolio_optimization.utils.tools import *

__all__ = ['FitnessType',
           'Assets',
           'Portfolio']


class FitnessType(Enum):
    MEAN_STD = 'mean_std'
    MEAN_DOWNSIDE_STD = 'mean_downside_std'
    MEAN_DOWNSIDE_STD_DRAWDOWN = 'mean_downside_std_drawdown'


class Assets:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.returns = preprocessing(prices=self.prices, to_array=True)
        self.mu = np.mean(self.returns, axis=1)
        self.cov = np.cov(self.returns)
        self.asset_nb, self.date_nb = self.returns.shape

    def __str__(self):
        return f'Assets (asset number: {self.asset_nb}, date number: {self.date_nb})'

    def __repr__(self):
        return str(self)


class Portfolio:
    def __init__(self, weights: np.ndarray, fitness_type: FitnessType, assets: Assets):
        # Pointer to Assets
        self.assets = assets
        if self.assets.asset_nb != len(weights):
            raise ValueError(f'weights should be of size {self.assets.asset_nb}')
        if not isinstance(weights, np.ndarray):
            raise TypeError(f'weights should be of type numpy.ndarray')
        self.fitness_type = fitness_type
        self.weights = weights
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
    def std(self):
        if self._std is None:
            self._std = np.sqrt(self.weights @ self.assets.cov @ self.weights)
        return self._std

    @property
    def downside_std(self):
        if self._downside_std is None:
            self._downside_std = self.returns[self.returns < 0].std(ddof=1)
        return self._downside_std

    @property
    def max_drawdown(self):
        if self._max_drawdown is None:
            self._max_drawdown = max_drawdown(self.prices)
        return self._max_drawdown

    @property
    def sharp_ratio(self):
        return self.mu / self.std

    @property
    def sortino_ratio(self):
        return self.mu / self.downside_std

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
            elif self.fitness_type == FitnessType.MEAN_DOWNSIDE_STD_DRAWDOWN:
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
