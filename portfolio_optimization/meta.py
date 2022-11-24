from enum import Enum

__all__ = ['AVG_TRADING_DAYS_PER_YEAR',
           'ZERO_THRESHOLD',
           'Metrics',
           'InvestmentType']

AVG_TRADING_DAYS_PER_YEAR = 255
ZERO_THRESHOLD = 1e-4


class InvestmentType(Enum):
    FULLY_INVESTED = 'fully_invested'
    MARKET_NEUTRAL = 'market_neutral'
    UNCONSTRAINED = 'unconstrained'


class Metrics(Enum):
    MEAN = 'mean'
    ANNUALIZED_MEAN = 'annualized_mean'
    STD = 'std'
    ANNUALIZED_STD = 'annualized_std'
    VARIANCE = 'variance'
    ANNUALIZED_VARIANCE = 'annualized_variance'
    DOWNSIDE_STD = 'downside_std'
    ANNUALIZED_DOWNSIDE_STD = 'annualized_downside_std'
    DOWNSIDE_VARIANCE = 'downside_variance'
    ANNUALIZED_DOWNSIDE_VARIANCE = 'annualized_downside_variance'
    MAX_DRAWDOWN = 'max_drawdown'
    CDAR = 'cdar'
    CVAR = 'cvar'
    SHARPE_RATIO = 'sharpe_ratio'
    SORTINO_RATIO = 'sortino_ratio'
    CALMAR_RATIO = 'calmar_ratio'
    CDAR_RATIO = 'cdar_ratio'
    CVAR_RATIO = 'cvar_ratio'

    @property
    def is_ration(self) -> bool:
        return self.value[-5:] == 'ratio'


