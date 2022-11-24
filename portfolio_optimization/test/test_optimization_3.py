import numpy as np

from portfolio_optimization import (Assets, load_prices, load_assets, Optimization, RiskMeasure, InvestmentType,
                                    EXAMPLE_PRICES_PATH, ObjectiveFunction, Portfolio, OptimizationError)


def is_close(a: float, b: float, precision: float = 1e-7):
    return abs(a - b) < precision


def get_assets() -> Assets:
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices = prices.iloc[:, 50:100].copy()
    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.90,
                         verbose=False)

    return assets


def test_mean_variance():
    precision = {
        RiskMeasure.VARIANCE: 1e-7,
        RiskMeasure.SEMI_VARIANCE: 1e-7,
        RiskMeasure.CVAR: 1e-4,
        RiskMeasure.CDAR: 1e-4
    }

    assets = get_assets()
    # previous_weights = np.random.randn(assets.asset_nb) / 10
    previous_weights = np.array([0.06663786, -0.02609581, -0.12200097, -0.03729676, -0.18604607,
                                 -0.09291357, -0.22839449, -0.08750029, 0.01262641, 0.08712638,
                                 -0.15731865, 0.14594815, 0.11637876, 0.02163102, 0.03458678,
                                 -0.1106219, -0.05892651, 0.05990245, -0.11428092, -0.06343284,
                                 0.0423514, 0.1394012, 0.00185886, 0.00799553, 0.02767036,
                                 -0.1706394, 0.07544119, -0.06341742, -0.0254911, -0.07081295,
                                 0.02034429, -0.03295023, 0.09833698, 0.0489829, -0.13253346])

    # transaction_costs = abs(np.random.randn(assets.asset_nb))/100
    transaction_costs = np.array([3.07368300e-03, 1.22914659e-02, 1.31012389e-02, 5.11069233e-03,
                                  3.14226164e-03, 1.38225267e-02, 1.01730423e-02, 1.60753223e-02,
                                  2.16640987e-04, 1.14058494e-02, 8.94785339e-03, 7.30764696e-03,
                                  1.82260135e-02, 2.00042452e-02, 8.56386327e-03, 4.53884918e-03,
                                  1.00539220e-02, 3.53354996e-04, 6.15081648e-03, 1.16504714e-02,
                                  5.66981399e-03, 1.33982849e-02, 2.77254069e-03, 5.52234266e-03,
                                  1.52447716e-05, 6.58091620e-03, 1.25069156e-02, 1.32262548e-02,
                                  7.73299012e-03, 5.38849221e-03, 1.51744779e-02, 5.22349873e-03,
                                  8.18506176e-03, 1.34491053e-02, 9.20145325e-03])

    for risk_measure in RiskMeasure:
        print(risk_measure)
        max_risk_arg = f'max_{risk_measure.value}'
        for weight_bounds in [(None, None), (0, None)]:
            for investment_type in InvestmentType:
                model = Optimization(assets=assets,
                                     investment_type=investment_type,
                                     weight_bounds=weight_bounds,
                                     previous_weights=previous_weights,
                                     transaction_costs=transaction_costs,
                                     investment_duration=assets.date_nb,
                                     solvers=['ECOS'])

                min_risk, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                           objective_function=ObjectiveFunction.MIN_RISK,
                                                           objective_values=True)

                p = Portfolio(assets=assets,
                              weights=w,
                              previous_weights=previous_weights,
                              transaction_costs=transaction_costs)
                assert is_close(getattr(p, risk_measure.value), min_risk, precision[risk_measure])
                min_risk_mean = p.mean

                risk = min_risk*1.2
                mean, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                       objective_function=ObjectiveFunction.MAX_RETURN,
                                                       objective_values=True,
                                                       **{max_risk_arg: risk})

                p = Portfolio(assets=assets,
                              weights=w,
                              previous_weights=previous_weights,
                              transaction_costs=transaction_costs)
                assert is_close(getattr(p, risk_measure.value), risk, precision[risk_measure])
                assert is_close(p.mean, mean)
                assert mean >= min_risk_mean - 1e-6

                w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                 objective_function=ObjectiveFunction.MIN_RISK,
                                                 **{max_risk_arg: risk})

                p = Portfolio(assets=assets,
                              weights=w,
                              previous_weights=previous_weights,
                              transaction_costs=transaction_costs)
                assert is_close(getattr(p, risk_measure.value), min_risk, precision[risk_measure])

                ret = mean
                risk, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                       objective_function=ObjectiveFunction.MIN_RISK,
                                                       min_return=ret,
                                                       objective_values=True)

                p = Portfolio(assets=assets,
                              weights=w,
                              previous_weights=previous_weights,
                              transaction_costs=transaction_costs)
                assert is_close(getattr(p, risk_measure.value), risk, precision[risk_measure])
                assert is_close(p.mean, mean, 1e-4)

                mean, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                       objective_function=ObjectiveFunction.MAX_RETURN,
                                                       objective_values=True,
                                                       **{max_risk_arg: risk})

                p = Portfolio(assets=assets,
                              weights=w,
                              previous_weights=previous_weights,
                              transaction_costs=transaction_costs)
                assert is_close(getattr(p, risk_measure.value), risk, precision[risk_measure])
                assert is_close(p.mean, mean)
