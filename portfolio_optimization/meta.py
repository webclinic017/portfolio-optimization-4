from enum import Enum

__all__ = ['AVG_TRADING_DAYS_PER_YEAR',
           'ZERO_THRESHOLD',
           'Metrics',
           'InvestmentType',
           'RiskMeasure',
           'ObjectiveFunction']

AVG_TRADING_DAYS_PER_YEAR = 255
ZERO_THRESHOLD = 1e-4


class InvestmentType(Enum):
    FULLY_INVESTED = 'fully_invested'
    MARKET_NEUTRAL = 'market_neutral'
    UNCONSTRAINED = 'unconstrained'


class RiskMeasure(Enum):
    VARIANCE = 'variance'
    SEMIVARIANCE = 'semivariance'
    CVAR = 'cvar'
    CDAR = 'cdar'
    MAD = 'mad'


class Metrics(Enum):
    MEAN = 'mean'
    ANNUALIZED_MEAN = 'annualized_mean'
    STD = 'std'
    ANNUALIZED_STD = 'annualized_std'
    VARIANCE = 'variance'
    ANNUALIZED_VARIANCE = 'annualized_variance'
    SEMISTD = 'semistd'
    ANNUALIZED_SEMISTD = 'annualized_semistd'
    SEMIVARIANCE = 'semivariance'
    ANNUALIZED_SEMIVARIANCE = 'annualized_semivariance'
    MAX_DRAWDOWN = 'max_drawdown'
    CDAR = 'cdar'
    CVAR = 'cvar'
    MAD = 'mad'
    SHARPE_RATIO = 'sharpe_ratio'
    SORTINO_RATIO = 'sortino_ratio'
    CALMAR_RATIO = 'calmar_ratio'
    CDAR_RATIO = 'cdar_ratio'
    CVAR_RATIO = 'cvar_ratio'

    @property
    def is_ration(self) -> bool:
        return self.value[-5:] == 'ratio'


class ObjectiveFunction(Enum):
    MIN_RISK = 'min_risk'
    MAX_RETURN = 'max_return'
    RATIO = 'ratio'
    UTILITY = 'utility'
