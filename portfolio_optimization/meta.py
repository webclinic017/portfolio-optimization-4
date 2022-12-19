from enum import Enum, auto

__all__ = ['AVG_TRADING_DAYS_PER_YEAR',
           'ZERO_THRESHOLD',
           'Perf',
           'RiskMeasure',
           'ObjectiveFunction',
           'Ratio',
           'RiskMeasureRatio',
           'RatioRiskMeasure',
           'Metrics',
           'MetricsValues']

AVG_TRADING_DAYS_PER_YEAR = 255
ZERO_THRESHOLD = 1e-4


class AutoEnum(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    @classmethod
    def has(cls, value):
        return value in cls._value2member_map_


class Perf(AutoEnum):
    MEAN = auto()


class RiskMeasure(AutoEnum):
    MAD = auto()
    FIRST_LOWER_PARTIAL_MOMENT = auto()
    VARIANCE = auto()
    STD = auto()
    SEMI_VARIANCE = auto()
    SEMI_STD= auto()
    KURTOSIS = auto()
    SEMI_KURTOSIS = auto()
    VALUE_AT_RISK = auto()
    CVAR = auto()
    ENTROPIC_RISK_MEASURE = auto()
    EVAR = auto()
    WORST_REALIZATION = auto()
    DAR = auto()
    CDAR = auto()
    MAX_DRAWDOWN = auto()
    AVG_DRAWDOWN = auto()
    EDAR = auto()
    ULCER_INDEX = auto()
    GINI_MEAN_DIFFERENCE = auto()

    def ratio(self):
        try:
            return RiskMeasureRatio[self]
        except KeyError:
            return Ratio(f'{self.value}_ratio')


class Ratio(AutoEnum):
    MAD_RATIO = auto()
    FIRST_LOWER_PARTIAL_MOMENT_RATIO = auto()
    SHARPE_RATIO = auto()
    SORTINO_RATIO = auto()
    KURTOSIS_RATIO = auto()
    SEMI_KURTOSIS_RATIO = auto()
    VALUE_AT_RISK_RATIO = auto()
    CVAR_RATIO = auto()
    ENTROPIC_RISK_MEASURE_RATIO = auto()
    EVAR_RATIO = auto()
    WORST_REALIZATION_RATIO = auto()
    DAR_RATIO = auto()
    CDAR_RATIO = auto()
    CALMAR_RATIO = auto()
    AVG_DRAWDOWN_RATIO = auto()
    EDAR_RATIO = auto()
    ULCER_INDEX_RATIO = auto()
    GINI_MEAN_DIFFERENCE_RATIO = auto()

    def risk_measure(self):
        try:
            return RatioRiskMeasure[self]
        except KeyError:
            return RiskMeasure(self.value.replace('_ratio', ''))


Metrics = {e for enu in [Perf, RiskMeasure, Ratio] for e in enu}
MetricsValues = {e.value: e for e in Metrics}

RiskMeasureRatio = {
    RiskMeasure.VARIANCE: Ratio.SHARPE_RATIO,
    RiskMeasure.STD: Ratio.SHARPE_RATIO,
    RiskMeasure.SEMI_VARIANCE: Ratio.SORTINO_RATIO,
    RiskMeasure.SEMI_STD: Ratio.SORTINO_RATIO,
    RiskMeasure.MAX_DRAWDOWN: Ratio.CALMAR_RATIO,
}

RatioRiskMeasure = {v: k for k, v in RiskMeasureRatio.items()}


class ObjectiveFunction(Enum):
    MIN_RISK = auto()
    MAX_RETURN = auto()
    RATIO = auto()
    UTILITY = auto()
