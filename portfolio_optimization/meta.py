from enum import Enum

__all__ = ['AVG_TRADING_DAYS_PER_YEAR',
           'ZERO_THRESHOLD',
           'InvestmentType']

AVG_TRADING_DAYS_PER_YEAR = 255
ZERO_THRESHOLD = 1e-4


class InvestmentType(Enum):
    FULLY_INVESTED = 'fully_invested'
    MARKET_NEUTRAL = 'market_neutral'
    UNCONSTRAINED = 'unconstrained'
