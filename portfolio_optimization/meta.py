from enum import Enum

__all__ = ['AVG_TRADING_DAYS_PER_YEAR',
           'ZERO_THRESHOLD',
           'Metrics',
           'InvestmentType',
           'FitnessType']

AVG_TRADING_DAYS_PER_YEAR = 255
ZERO_THRESHOLD = 1e-4


class InvestmentType(Enum):
    FULLY_INVESTED = 'fully_invested'
    MARKET_NEUTRAL = 'market_neutral'
    UNCONSTRAINED = 'unconstrained'


class Metrics(Enum):
    MEAN = 'mean'
    STD = 'std'
    VARIANCE = 'variance'
    DOWNSIDE_STD = 'downside_std'
    DOWNSIDE_VARIANCE = 'downside_variance'
    ANNUALIZED_MEAN = 'annualized_mean'
    ANNUALIZED_STD = 'annualized_std'
    ANNUALIZED_DOWNSIDE_STD = 'annualized_downside_std'
    MAX_DRAWDOWN = 'max_drawdown'
    CDAR_95 = 'cdar_95'
    CVAR_95 = 'cvar_95'
    SHARPE_RATIO = 'sharpe_ratio'
    SORTINO_RATIO = 'sortino_ratio'
    CALMAR_RATIO = 'calmar_ratio'
    CDAR_95_RATIO = 'cdar_95_ratio'
    CVAR_95_RATIO = 'cvar_95_ratio'


class FitnessType(Enum):
    MEAN_STD = (Metrics.MEAN, Metrics.STD)
    MEAN_DOWNSIDE_STD = (Metrics.MEAN, Metrics.DOWNSIDE_STD)
    MEAN_DOWNSIDE_STD_MAX_DRAWDOWN = (Metrics.MEAN, Metrics.DOWNSIDE_STD, Metrics.MAX_DRAWDOWN)
