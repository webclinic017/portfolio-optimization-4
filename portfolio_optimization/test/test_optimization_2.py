import numpy as np
import pandas as pd
from pathlib import Path
import riskfolio as rp
import time

from portfolio_optimization import *


def t():
    stocks = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'NBL', 'APA', 'MMC', 'JPM', 'ZION']
    prices = load_prices(file=TEST_PRICES_PATH)[stocks][-200:]
    assets = load_assets(prices=prices, verbose=False)
    population = Population()
    Y = prices.pct_change().dropna().iloc[-200:]

    # val=0.01
    port = rp.Portfolio(returns=Y, sht=True, budget=0.5)
    port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
    m = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = 'CVaR'  # Risk measure used, this time will be variance
    obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True  # Use historical scenarios for risk measures that depend on scenarios
    rf = 0  # Risk free rate
    l = 0  # Risk aversion factor, only useful when obj is 'Utility'
    w = port.optimization(model=m, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    d = w.to_dict()['weights']
    w1 = np.array([d[c] for c in stocks])
    sum(w1)
    sum(abs(w1))
    sum(np.maximum(w1, 0))
    sum(np.minimum(w1, 0))

    model = Optimization(assets=assets,
                         min_weights=-1,
                         max_weights=1,
                         max_short=0.2,
                         max_long=1,
                         budget=0.5)
    r , w2 = model.mean_risk_optimization(risk_measure=RiskMeasure.CVAR,
                                      objective_function=ObjectiveFunction.RATIO,
                                      objective_values=True)
    sum(w2)
    sum(abs(w2))
    sum(np.maximum(w2, 0))
    sum(np.minimum(w2, 0))
    print(Portfolio(assets=assets,weights=w1).sharpe_ratio)
    print(Portfolio(assets=assets,weights=w2).sharpe_ratio)
    print(Portfolio(assets=assets,weights=w2).mean)
    print(Portfolio(assets=assets,weights=w2).cvar)


    np.testing.assert_array_almost_equal(w1, w2, decimal=3)

    for upperdev in np.linspace(0.008, 0.015, num=10):
        for upperCDaR in np.linspace(0.03, 0.11, num=10):
            port = rp.Portfolio(returns=Y, upperdev=upperdev, upperCDaR=upperCDaR)
            port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
            m = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'MV'  # Risk measure used, this time will be variance
            obj = 'MaxRet'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True  # Use historical scenarios for risk measures that depend on scenarios
            rf = 0  # Risk free rate
            l = 0  # Risk aversion factor, only useful when obj is 'Utility'
            w = port.optimization(model=m, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            d = w.to_dict()['weights']
            w1 = np.array([d[c] for c in stocks])
            w2 = model.optimize(risk_measure=RiskMeasure.VARIANCE,
                                objective_function=ObjectiveFunction.MAX_RETURN,
                                max_variance=upperdev ** 2,
                                max_cdar=upperCDaR)
            np.testing.assert_array_almost_equal(w1, w2, decimal=3)

    for uppersdev in np.linspace(0.008, 0.015, num=10):
        for upperCDaR in np.linspace(0.03, 0.11, num=10):
            port = rp.Portfolio(returns=Y, uppersdev=uppersdev, upperCDaR=upperCDaR)
            port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
            m = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'SMV'  # Risk measure used, this time will be variance
            obj = 'MaxRet'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True  # Use historical scenarios for risk measures that depend on scenarios
            rf = 0  # Risk free rate
            l = 0  # Risk aversion factor, only useful when obj is 'Utility'
            w = port.optimization(model=m, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            d = w.to_dict()['weights']
            w1 = np.array([d[c] for c in stocks])
            w2 = model.optimize(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                objective_function=ObjectiveFunction.MAX_RETURN,
                                max_semivariance=uppersdev ** 2,
                                max_cdar=upperCDaR)
            np.testing.assert_array_almost_equal(w1, w2, decimal=3)

    s = time.time()
    model = Optimization(assets=assets, weight_bounds=(0, None))
    w2 = model.optimize(risk_measure=RiskMeasure.VARIANCE,
                        objective_function=ObjectiveFunction.MAX_RETURN,
                        max_semivariance=uppersdev ** 2,
                        max_cdar=upperCDaR)
    e = time.time()
    print((e - s) * 1000)

    s = time.time()
    port = rp.Portfolio(returns=Y, upperdev=uppersdev, upperCDaR=upperCDaR)
    port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
    m = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = 'SMV'  # Risk measure used, this time will be variance
    obj = 'MaxRet'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True  # Use historical scenarios for risk measures that depend on scenarios
    rf = 0  # Risk free rate
    l = 0  # Risk aversion factor, only useful when obj is 'Utility'
    w = port.optimization(model=m, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    e = time.time()
    print((e - s) * 1000)

    population.plot_metrics(x=Metrics.ANNUALIZED_STD, y=Metrics.CDAR, z=Metrics.ANNUALIZED_MEAN, to_surface=True)

    self = model


def get_assets() -> Assets:
    stocks = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'NBL', 'APA', 'MMC', 'JPM', 'ZION']
    # benchmark = ['SPY']
    prices = load_prices(file=TEST_PRICES_PATH)[stocks][-200:]
    assets = load_assets(prices=prices, verbose=False)
    return assets


def test_minimum_risk():
    ref = pd.read_csv(Path(TEST_FOLDER, 'data', 'Classic_MinRisk.csv'))
    assets = get_assets()
    model = Optimization(assets=assets, weight_bounds=(0, None))

    minimum, weight = model.minimum_variance()
    ref_weight = ref['MV'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)

    minimum, weight = model.minimum_variance2()
    ref_weight = ref['MV'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)

    minimum, weight = model.mean_variance2(optimization_results=True)
    ref_weight = ref['MV'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)

    minimum, weight = model.mean_variance2(target_return=0.0015500395821130648, optimization_results=True)
    ref_weight = ref['MV'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)

    import time

    s = time.time()
    w = []
    for i in range(100):
        minimum, weight = model.minimum_variance2()
        w.append(minimum)
    e = time.time()
    print((e - s) * 1000)
    np.mean(w)
    np.std(w)

    """
    5.386013418256054e-05
2.0328790734103208e-20
"""

    minimum, weight = model.minimum_cvar(beta=0.95)
    ref_weight = ref['CVaR'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)

    minimum, weight = model.minimum_cdar(beta=0.95)
    ref_weight = ref['CDaR'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)

    minimum, weight = model.minimum_semi_variance()
    ref_weight = ref['MSV'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)


def test_maximum_ratio():
    ref = pd.read_csv(Path(TEST_FOLDER, 'data', 'Classic_Sharpe.csv'))
    assets = get_assets()
    model = Optimization(assets=assets, weight_bounds=(0, None))

    weight = model.maximum_sharpe()
    ref_weight = ref['MV'].to_numpy()
    np.testing.assert_array_almost_equal(weight, ref_weight, decimal=2)
