from enum import Enum, auto

__all__ = ['AVG_TRADING_DAYS_PER_YEAR',
           'ZERO_THRESHOLD',
           'Metric',
           'RiskMeasure',
           'ObjectiveFunction',
           'Ratio',
           'RiskMeasureRatio',
           'RatioRiskMeasure']

AVG_TRADING_DAYS_PER_YEAR = 255
ZERO_THRESHOLD = 1e-4


class AutoEnum(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class RiskMeasure(AutoEnum):
    MAD = auto()
    FIRST_LOWER_PARTIAL_MOMENT = auto()
    VARIANCE = auto()
    SEMI_VARIANCE = auto()
    # KURTOSIS = auto()
    # SEMI_KURTOSIS = auto()
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
        return RatioRiskMeasure.get(self, Ratio(f'{self.value}_ratio'))


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
        return RatioRiskMeasure.get(self, RiskMeasure(self.value.replace('_ratio', '')))


RiskMeasureRatio = {
    RiskMeasure.VARIANCE: Ratio.SHARPE_RATIO,
    RiskMeasure.SEMI_VARIANCE: Ratio.SORTINO_RATIO,
    RiskMeasure.MAX_DRAWDOWN: Ratio.CALMAR_RATIO,
}

RatioRiskMeasure = {v: k for k, v in RiskMeasureRatio.items()}

_metrics = {'MEAN': 'mean'}
_metrics.update({e.name: e.value for e in RiskMeasure})
_metrics.update({e.name: e.value for e in Ratio})
_metrics.update({'STD': 'std',
                 'SEMI_STD': 'semi_std',
                 'KURTOSIS': 'kurtosis',
                 'SEMI_KURTOSIS': 'semi_kurtosis'})

Metric = Enum('Metric', _metrics)


class ObjectiveFunction(Enum):
    MIN_RISK = auto()
    MAX_RETURN = auto()
    RATIO = auto()
    UTILITY = auto()
