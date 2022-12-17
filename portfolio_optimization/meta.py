from enum import Enum

__all__ = ['AVG_TRADING_DAYS_PER_YEAR',
           'ZERO_THRESHOLD',
           'Metrics',
           'InvestmentType',
           'RiskMeasure',
           'ObjectiveFunction',
           'Ratios']

AVG_TRADING_DAYS_PER_YEAR = 255
ZERO_THRESHOLD = 1e-4


class InvestmentType(Enum):
    FULLY_INVESTED = 'fully_invested'
    MARKET_NEUTRAL = 'market_neutral'
    UNCONSTRAINED = 'unconstrained'


class RiskMeasure(Enum):
    MAD = 'mad'
    VARIANCE = 'variance'
    SEMI_VARIANCE = 'semi_variance'
    CVAR = 'cvar'
    WORST_REALISATION = 'worst_realisation'
    LOWER_PARTIAL_MOMENT = 'lower_partial_moment'
    MAX_DRAWDOWN = 'max_drawdown'
    AVG_DRAWDOWN = 'avg_drawdown'
    CDAR = 'cdar'
    ULCER_INDEX = 'ulcer_index'


class Ratios(Enum):
    SHARPE_RATIO = 'sharpe_ratio'
    SORTINO_RATIO = 'sortino_ratio'
    CALMAR_RATIO = 'calmar_ratio'
    CDAR_RATIO = 'cdar_ratio'
    CVAR_RATIO = 'cvar_ratio'


_metrics = {'MEAN': 'mean'}
_metrics.update({e.name: e.value for e in RiskMeasure})
_metrics.update({e.name: e.value for e in Ratios})
_metrics.update({'STD': 'std',
                 'SEMI_STD': 'semi_std',
                 'KURTOSIS': 'kurtosis',
                 'SEMI_KURTOSIS': 'semi_kurtosis'})

Metrics = Enum('Metrics', _metrics)


class ObjectiveFunction(Enum):
    MIN_RISK = 'min_risk'
    MAX_RETURN = 'max_return'
    RATIO = 'ratio'
    UTILITY = 'utility'
