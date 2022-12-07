import numpy as np

from portfolio_optimization import (Assets, load_prices, load_assets, Optimization, RiskMeasure, InvestmentType,
                                    EXAMPLE_PRICES_PATH, ObjectiveFunction, Portfolio, OptimizationError)
from portfolio_optimization.optimization.group_constraints import group_constraints_to_matrix


def is_close(a: float, b: float, precision: float = 1e-7):
    return abs(a - b) < precision


def get_assets() -> Assets:
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices = prices.iloc[:, 50:80].copy()
    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.90,
                         verbose=False)

    return assets


def test_mean_risk_optimization():
    precision = {
        RiskMeasure.VARIANCE: 1e-7,
        RiskMeasure.SEMIVARIANCE: 1e-7,
        RiskMeasure.CVAR: 1e-4,
        RiskMeasure.CDAR: 1e-4
    }

    assets = get_assets()
    # previous_weights = np.random.randn(assets.asset_nb) / 10
    previous_weights = np.array([0.06663786, -0.02609581, -0.12200097, -0.03729676, -0.18604607,
                                 -0.09291357, -0.22839449, -0.08750029, 0.01262641, 0.08712638,
                                 -0.15731865, 0.14594815, 0.11637876, 0.02163102, 0.03458678,
                                 -0.1106219, -0.05892651, 0.05990245])

    # transaction_costs = abs(np.random.randn(assets.asset_nb))/100
    transaction_costs = np.array([3.07368300e-03, 1.22914659e-02, 1.31012389e-02, 5.11069233e-03,
                                  3.14226164e-03, 1.38225267e-02, 1.01730423e-02, 1.60753223e-02,
                                  2.16640987e-04, 1.14058494e-02, 8.94785339e-03, 7.30764696e-03,
                                  1.82260135e-02, 2.00042452e-02, 8.56386327e-03, 1.38225267e-02,
                                  1.01730423e-02, 1.01730423e-02])

    asset_groups = [['Equity'] * 3 + ['Fund'] * 3 + ['Bond'] * 12,
                    ['US'] * 2 + ['Europe'] * 6 + ['Japan'] * 10]

    group_constraints = ['Equity <= 0.5 * Bond',
                         'US >= 0.1',
                         'Europe >= 0.5 * Fund',
                         'Japan <= 1']

    left_inequality, right_inequality = group_constraints_to_matrix(groups=np.array(asset_groups),
                                                                    constraints=group_constraints)

    params = [dict(min_weights=-1,
                   max_weights=1,
                   previous_weights=previous_weights,
                   transaction_costs=transaction_costs,
                   investment_duration=assets.date_nb),
              dict(min_weights=0,
                   max_weights=None,
                   budget=0.5,
                   previous_weights=previous_weights,
                   transaction_costs=transaction_costs,
                   investment_duration=assets.date_nb),
              dict(min_weights=-0.2,
                   max_weights=10,
                   budget=None,
                   min_budget=-1,
                   max_budget=1,
                   previous_weights=previous_weights,
                   transaction_costs=transaction_costs,
                   investment_duration=assets.date_nb),
              dict(min_weights=-1,
                   max_weights=1,
                   min_budget=-1,
                   budget=None,
                   max_budget=1,
                   max_short=0.5,
                   max_long=2,
                   previous_weights=previous_weights,
                   transaction_costs=transaction_costs,
                   investment_duration=assets.date_nb),
              dict(min_weights=-1,
                   max_weights=1,
                   max_short=0.5,
                   max_long=2,
                   previous_weights=previous_weights,
                   transaction_costs=transaction_costs,
                   investment_duration=assets.date_nb,
                   asset_groups=asset_groups,
                   group_constraints=group_constraints),
              dict(min_weights=-1,
                   max_weights=1,
                   max_short=0.5,
                   max_long=2,
                   previous_weights=previous_weights,
                   transaction_costs=transaction_costs,
                   investment_duration=assets.date_nb,
                   left_inequality=left_inequality,
                   right_inequality=right_inequality)
              ]

    for risk_measure in RiskMeasure:
        print(risk_measure)
        max_risk_arg = f'max_{risk_measure.value}'

        for param in params:
            model = Optimization(assets=assets,
                                 solvers=['ECOS'],
                                 **param)

            if risk_measure in [RiskMeasure.CDAR]:
                model.update(scale=0.01)

            min_risk, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                       objective_function=ObjectiveFunction.MIN_RISK,
                                                       objective_values=True)

            p = Portfolio(assets=assets,
                          weights=w,
                          previous_weights=previous_weights,
                          transaction_costs=transaction_costs)
            assert is_close(getattr(p, risk_measure.value), min_risk, precision[risk_measure])
            min_risk_mean = p.mean

            risk = min_risk * 1.2
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

            risk, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                   objective_function=ObjectiveFunction.MIN_RISK,
                                                   min_return=mean,
                                                   objective_values=True)

            p = Portfolio(assets=assets,
                          weights=w,
                          previous_weights=previous_weights,
                          transaction_costs=transaction_costs)
            assert is_close(getattr(p, risk_measure.value), risk, precision[risk_measure])
            assert is_close(p.mean, mean, max(precision[risk_measure], 1e-5))

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

            # utility
            gamma = 3
            utility, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                      objective_function=ObjectiveFunction.UTILITY,
                                                      gamma=gamma,
                                                      objective_values=True)
            p = Portfolio(assets=assets,
                          weights=w,
                          previous_weights=previous_weights,
                          transaction_costs=transaction_costs)
            p_utility = p.mean - gamma * getattr(p, risk_measure.value)
            assert is_close(p_utility, utility, precision[risk_measure])

            utility, w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                      objective_function=ObjectiveFunction.UTILITY,
                                                      gamma=gamma,
                                                      objective_values=True,
                                                      **{max_risk_arg: risk})
            p = Portfolio(assets=assets,
                          weights=w,
                          previous_weights=previous_weights,
                          transaction_costs=transaction_costs)
            p_utility = p.mean - gamma * getattr(p, risk_measure.value)
            assert is_close(p_utility, utility, precision[risk_measure])

            # ratio
            if risk_measure in [RiskMeasure.CDAR]:
                model.update(scale=10)
            else:
                model.update(scale=1000)
            model.update(transaction_costs=0)
            (mean, risk), w = model.mean_risk_optimization(risk_measure=risk_measure,
                                                           objective_function=ObjectiveFunction.RATIO,
                                                           objective_values=True)
            p = Portfolio(assets=assets, weights=w)
            assert is_close(mean, p.mean, precision[risk_measure])
            assert is_close(risk, getattr(p, risk_measure.value), precision[risk_measure])


def test_minimum_risk_methods():
    assets = get_assets()
    model = Optimization(assets=assets, solvers=['ECOS'])
    for risk_measure in RiskMeasure:
        func = getattr(model, f'minimum_{risk_measure.value}')
        w = func()
        assert not np.all(np.isnan(w))
        o, w = func(objective_values=True)
        assert o
        assert not np.all(np.isnan(w))


def test_mean_variance():
    assets = get_assets()

    model = Optimization(assets=assets, solvers=['ECOS'], min_weights=0)

    min_risk, w = model.mean_variance(objective_values=True)

    risk = min_risk * 3
    ret, w = model.mean_variance(target_variance=risk, objective_values=True)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(risk, p.variance, 1e-8)

    w = model.mean_variance(target_return=ret)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(ret, p.mean, 1e-8)

    w = model.mean_variance(target_variance=risk, l1_coef=0.1, l2_coef=0.1)
    assert np.all(w)

    risks = [risk, risk * 1.5]
    o, w = model.mean_variance(target_variance=risks, objective_values=True)
    assert o.shape == (2,)
    assert w.shape == (2, assets.asset_nb)
    p0 = Portfolio(assets=assets, weights=w[0])
    p1 = Portfolio(assets=assets, weights=w[1])
    assert is_close(risks[0], p0.variance, 1e-8)
    assert is_close(risks[1], p1.variance, 1e-8)

    o, w = model.mean_variance(population_size=5, objective_values=True)
    assert o.shape == (5,)
    assert w.shape == (5, assets.asset_nb)
    p = Portfolio(assets=assets, weights=w[0])
    for i in range(1, 5):
        pi = Portfolio(assets=assets, weights=w[i])
        assert pi.variance > p.variance
        p = p1


def test_maximum_sharpe_ratio():
    assets = get_assets()

    model = Optimization(assets=assets, solvers=['ECOS'], min_weights=0)

    min_ratio, w = model.maximum_sharpe_ratio(objective_values=True)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(min_ratio[0] / np.sqrt(min_ratio[1]) * 255 / np.sqrt(255), p.sharpe_ratio, 1e-7)


def test_mean_semivariance():
    assets = get_assets()

    model = Optimization(assets=assets, solvers=['ECOS'], min_weights=0)

    min_risk, w = model.mean_semivariance(objective_values=True)

    risk = min_risk * 3
    ret, w = model.mean_semivariance(target_semivariance=risk, objective_values=True)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(risk, p.semivariance, 1e-8)

    w = model.mean_semivariance(target_semivariance=risk, min_acceptable_returns=0)
    p = Portfolio(assets=assets, weights=w, min_acceptable_return=0)
    assert is_close(risk, p.semivariance, 1e-8)

    w = model.mean_semivariance(target_return=ret)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(ret, p.mean, 1e-8)

    w = model.mean_semivariance(target_semivariance=risk, l1_coef=0.1, l2_coef=0.1)
    assert np.all(w)

    risks = [risk, risk * 1.5]
    o, w = model.mean_semivariance(target_semivariance=risks, objective_values=True)
    assert o.shape == (2,)
    assert w.shape == (2, assets.asset_nb)
    p0 = Portfolio(assets=assets, weights=w[0])
    p1 = Portfolio(assets=assets, weights=w[1])
    assert is_close(risks[0], p0.semivariance, 1e-8)
    assert is_close(risks[1], p1.semivariance, 1e-8)

    o, w = model.mean_variance(population_size=5, objective_values=True)
    assert o.shape == (5,)
    assert w.shape == (5, assets.asset_nb)
    p = Portfolio(assets=assets, weights=w[0])
    for i in range(1, 5):
        pi = Portfolio(assets=assets, weights=w[i])
        assert pi.semivariance > p.semivariance
        p = p1


def test_mean_cvar():
    assets = get_assets()

    model = Optimization(assets=assets, solvers=['ECOS'], min_weights=0)

    min_risk, w = model.mean_cvar(objective_values=True)

    risk = min_risk * 2
    ret, w = model.mean_cvar(target_cvar=risk, objective_values=True)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(risk, p.cvar, 1e-4)

    w = model.mean_cvar(target_cvar=risk, cvar_beta=0.5)
    p = Portfolio(assets=assets, weights=w, cvar_beta=0.5)
    assert is_close(risk, p.cvar, 1e-4)

    w = model.mean_cvar(target_return=ret)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(ret, p.mean, 1e-4)

    w = model.mean_cvar(target_cvar=risk, l1_coef=0.1, l2_coef=0.1)
    assert np.all(w)

    risks = [risk, risk * 1.5]
    o, w = model.mean_cvar(target_cvar=risks, objective_values=True)
    assert o.shape == (2,)
    assert w.shape == (2, assets.asset_nb)
    p0 = Portfolio(assets=assets, weights=w[0])
    p1 = Portfolio(assets=assets, weights=w[1])
    assert is_close(risks[0], p0.cvar, 1e-4)
    assert is_close(risks[1], p1.cvar, 1e-4)

    o, w = model.mean_variance(population_size=5, objective_values=True)
    assert o.shape == (5,)
    assert w.shape == (5, assets.asset_nb)
    p = Portfolio(assets=assets, weights=w[0])
    for i in range(1, 5):
        pi = Portfolio(assets=assets, weights=w[i])
        assert pi.cvar > p.cvar
        p = p1


def test_mean_cdar():
    assets = get_assets()

    model = Optimization(assets=assets, solvers=['ECOS'], min_weights=0)

    min_risk, w = model.mean_cdar(objective_values=True)

    risk = min_risk * 2
    ret, w = model.mean_cdar(target_cdar=risk, objective_values=True)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(risk, p.cdar, 1e-8)

    w = model.mean_cdar(target_cdar=risk, cdar_beta=0.5)
    p = Portfolio(assets=assets, weights=w, cdar_beta=0.5)
    assert is_close(risk, p.cdar, 1e-8)

    w = model.mean_cdar(target_return=ret)
    p = Portfolio(assets=assets, weights=w)
    assert is_close(ret, p.mean, 1e-8)

    w = model.mean_cdar(target_cdar=risk, l1_coef=0.1, l2_coef=0.1)
    assert np.all(w)

    risks = [risk, risk * 1.5]
    o, w = model.mean_cdar(target_cdar=risks, objective_values=True)
    assert o.shape == (2,)
    assert w.shape == (2, assets.asset_nb)
    p0 = Portfolio(assets=assets, weights=w[0])
    p1 = Portfolio(assets=assets, weights=w[1])
    assert is_close(risks[0], p0.cdar, 1e-8)
    assert is_close(risks[1], p1.cdar, 1e-8)

    o, w = model.mean_variance(population_size=5, objective_values=True)
    assert o.shape == (5,)
    assert w.shape == (5, assets.asset_nb)
    p = Portfolio(assets=assets, weights=w[0])
    for i in range(1, 5):
        pi = Portfolio(assets=assets, weights=w[i])
        assert pi.cdar > p.cdar
        p = p1
