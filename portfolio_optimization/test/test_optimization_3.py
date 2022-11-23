import numpy as np

from portfolio_optimization import *


def is_close(a: float, b: float):
    return abs(a - b) < 1e-7


def get_assets() -> Assets:
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices = prices.iloc[:, :100].copy()
    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.99,
                         verbose=False)

    return assets


def test_mean_variance():
    assets = get_assets()
    weight_bounds = (0, None)
    previous_weights = np.random.randn(assets.asset_nb)
    transaction_costs = abs(np.random.randn(assets.asset_nb)) / 100
    investment_type = InvestmentType.FULLY_INVESTED
    model = Optimization(assets=assets,
                         investment_type=investment_type,
                         weight_bounds=weight_bounds,
                         previous_weights=previous_weights,
                         transaction_costs=transaction_costs,
                         investment_duration=assets.date_nb)

    mean, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                           objective_function=ObjectiveFunction.MAX_RETURN,
                                           max_variance=0.1,
                                           objective_values=True)

    p = Portfolio(assets=assets,
                  weights=w,
                  previous_weights=previous_weights,
                  transaction_costs=transaction_costs)

    assert is_close(p.mean, mean)

    for weight_bounds in [(None, None), (0, None)]:
        for investment_type in InvestmentType:
            model = Optimization(assets=assets,
                                 investment_type=investment_type,
                                 weight_bounds=weight_bounds,
                                 previous_weights=previous_weights,
                                 transaction_costs=transaction_costs,
                                 investment_duration=assets.date_nb)

            min_variance, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                                           objective_function=ObjectiveFunction.MIN_RISK,
                                                           objective_values=True)

            p = Portfolio(assets=assets,
                          weights=w,
                          previous_weights=previous_weights,
                          transaction_costs=transaction_costs)
            assert is_close(p.variance, min_variance)

            var = min_variance +1e-4
            w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                             objective_function=ObjectiveFunction.MAX_RETURN,
                                             max_variance=var)

            p = Portfolio(assets=assets,
                          weights=w,
                          previous_weights=previous_weights,
                          transaction_costs=transaction_costs)
            assert is_close(p.variance, var)

            w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                             objective_function=ObjectiveFunction.MIN_RISK,
                                             max_variance=min_variance)

            p = Portfolio(assets=assets,
                          weights=w,
                          previous_weights=previous_weights,
                          transaction_costs=transaction_costs)
            assert is_close(p.variance, min_variance)

            ret = abs(p.mean + 0.001) * 3
            try:
                variance, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                                           objective_function=ObjectiveFunction.MIN_RISK,
                                                           min_return=ret,
                                                           objective_values=True)

                p = Portfolio(assets=assets,
                              weights=w,
                              previous_weights=previous_weights,
                              transaction_costs=transaction_costs)
                assert is_close(p.variance, variance)
                assert is_close(p.mean, ret)

                var = variance
                mean, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                                       objective_function=ObjectiveFunction.MAX_RETURN,
                                                       max_variance=var,
                                                       objective_values=True)

                p = Portfolio(assets=assets,
                              weights=w,
                              previous_weights=previous_weights,
                              transaction_costs=transaction_costs)
                assert is_close(p.variance, var)
                assert is_close(p.mean, mean)
            except OptimizationError:
                pass


def test_mean_semivariance():
    assets = get_assets()
    weight_bounds = (0, None)
    investment_type = InvestmentType.FULLY_INVESTED
    model = Optimization(assets=assets,
                         investment_type=investment_type,
                         weight_bounds=weight_bounds,
                         transaction_costs=0.01,
                         investment_duration=255)

    min_variance, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                                   objective_function=ObjectiveFunction.MIN_RISK,
                                                   objective_values=True)

    for weight_bounds in [(None, None), (0, None)]:
        for investment_type in InvestmentType:
            model = Optimization(assets=assets,
                                 investment_type=investment_type,
                                 weight_bounds=weight_bounds)

            min_variance, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                                           objective_function=ObjectiveFunction.MIN_RISK,
                                                           objective_values=True)

            p = Portfolio(assets=assets, weights=w)
            assert is_close(p.variance, min_variance)

            var = min_variance + 1e-8
            w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                             objective_function=ObjectiveFunction.MAX_RETURN,
                                             max_variance=var)

            p = Portfolio(assets=assets, weights=w)
            assert is_close(p.variance, var)

            w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                             objective_function=ObjectiveFunction.MIN_RISK,
                                             max_variance=min_variance)

            p = Portfolio(assets=assets, weights=w)
            assert is_close(p.variance, min_variance)

            ret = abs(p.mean + 0.001) * 3
            try:
                variance, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                                           objective_function=ObjectiveFunction.MIN_RISK,
                                                           min_return=ret,
                                                           objective_values=True)

                p = Portfolio(assets=assets, weights=w)
                assert is_close(p.variance, variance)
                assert is_close(p.mean, ret)

                var = variance
                mean, w = model.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                                       objective_function=ObjectiveFunction.MAX_RETURN,
                                                       max_variance=var,
                                                       objective_values=True)

                p = Portfolio(assets=assets, weights=w)
                assert is_close(p.variance, var)
                assert is_close(p.mean, mean)
            except OptimizationError:
                pass


def test_inverse_volatility():
    assets = get_assets()
    model = Optimization(assets=assets)
    weights = model.inverse_volatility()

    assert abs(sum(weights) - 1) < 1e-10
    w = 1 / assets.std
    w = w / sum(w)
    assert abs(weights - w).sum() < 1e-10


def test_equal_weighted():
    assets = get_assets()
    model = Optimization(assets=assets)
    weights = model.equal_weighted()

    assert abs(sum(weights) - 1) < 1e-10
    w = 1 / assets.asset_nb
    assert abs(weights - w).sum() < 1e-10


def test_random():
    assets = get_assets()
    model = Optimization(assets=assets)
    weights = model.random()

    assert abs(sum(weights) - 1) < 1e-10


def minimum_testing(method_name: str,
                    portfolio_target_name: str,
                    threshold: float,
                    **kwargs):
    minimum_method_name = method_name.replace('mean', 'minimum')
    assets = get_assets()
    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    min_func = getattr(model, minimum_method_name)
    # Population size
    min_risk, weights = min_func()
    assert isinstance(min_risk, float)
    assert min_risk > 0
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (assets.asset_nb,)
    assert not np.isnan(weights).any()

    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(getattr(portfolio, portfolio_target_name) - min_risk) < threshold

    mean_func = getattr(model, method_name)
    target_name = method_name.replace('mean', 'target')
    weights_2 = mean_func(**{target_name: min_risk})
    portfolio_2 = Portfolio(weights=weights_2,
                            assets=assets)
    assert abs(getattr(portfolio_2, portfolio_target_name) - min_risk) < threshold

    weights_3 = mean_func(**{target_name: min_risk / 2})
    if not np.isnan(weights_3).all():
        portfolio_3 = Portfolio(weights=weights_2,
                                assets=assets)
        assert abs(getattr(portfolio_3, portfolio_target_name) - min_risk) < threshold


def population_testing(method_name: str, **kwargs):
    assets = get_assets()
    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    func = getattr(model, method_name)
    # Population size
    portfolios_weights = func(population_size=30)
    assert isinstance(portfolios_weights, np.ndarray)
    assert portfolios_weights.shape == (30, assets.asset_nb)
    assert not np.isnan(portfolios_weights).any()


def investment_type_testing(method_name: str,
                            target: float,
                            **kwargs):
    target_name = method_name.replace('mean', 'target')
    assets = get_assets()

    # Fully invested and no short selling
    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, 10))
    func = getattr(model, method_name)
    weights = func(**{target_name: target})
    assert abs(sum(weights) - 1) < 1e-10
    assert np.all(weights >= 0)

    # Fully invested and short selling
    model.update(weight_bounds=(None, None))
    weights = func(**{target_name: target})
    assert abs(sum(weights) - 1) < 1e-10
    assert not np.all(weights >= 0)

    # Fully invested and short selling with weights between -20% and 30%
    lower = -0.2
    upper = 0.3
    model.update(weight_bounds=(lower, upper))
    weights = func(**{target_name: target})
    assert abs(sum(weights) - 1) < 1e-10
    assert np.all(weights >= lower) and np.all(weights <= upper)

    # Market neutral with short selling
    model.update(investment_type=InvestmentType.MARKET_NEUTRAL,
                 weight_bounds=(None, None))
    weights = func(**{target_name: target})
    assert abs(sum(weights)) < 1e-10
    assert sum(abs(weights)) > 1
    assert not np.all(weights >= 0)

    # Market neutral with no short selling
    try:
        model.update(investment_type=InvestmentType.MARKET_NEUTRAL,
                     weight_bounds=(0.1, None))
        raise
    except ValueError:
        pass

    # UNCONSTRAINED
    model.update(investment_type=InvestmentType.UNCONSTRAINED,
                 weight_bounds=(None, None))
    weights = func(**{target_name: target})
    assert abs(sum(weights) - 1) > 1e-5
    assert abs(sum(weights)) > 1e-5


def costs_testing(method_name: str,
                  target: float,
                  portfolio_target_name: str,
                  threshold: float):
    target_name = method_name.replace('mean', 'target')
    assets = get_assets()
    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    func = getattr(model, method_name)
    # Ref with no costs
    weights = func(**{target_name: target})
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')
    assert abs(getattr(portfolio_ref, portfolio_target_name) - target) < threshold

    # uniform costs for all assets and empty prev_weight --> no impact on weights
    model.update(costs=0.1,
                 prev_w=None,
                 investment_duration_in_days=255)
    weights = func(**{target_name: target})
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(getattr(portfolio, portfolio_target_name) - target) < threshold
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 5e-2

    # uniform costs for all assets and uniform prev_weight --> no impact on weights
    model.update(prev_w=np.ones(assets.asset_nb))
    weights = func(**{target_name: target})
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(getattr(portfolio, portfolio_target_name) - target) < threshold
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 5e-2

    # costs on top two invested assets and uniform prev_weight --> impact on the two invested assets weight
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.5}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=None)
    weights = func(**{target_name: target})
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(getattr(portfolio, portfolio_target_name) - target) < threshold
    assert asset_1 not in portfolio.composition.index
    assert asset_2 not in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 5e-2

    # costs and identical prev_weight on top two invested assets --> the top two assets weights stay > 0
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.5}
    prev_weights = {asset_1: portfolio_ref.get_weight(asset_name=asset_1),
                    asset_2: portfolio_ref.get_weight(asset_name=asset_2)}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = func(**{target_name: target})
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(getattr(portfolio, portfolio_target_name) - target) < threshold
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 5e-2

    # identical costs on all assets and large prev_weight on top two invested assets
    # --> the top two assets weights become larger
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    prev_weights = {asset_1: 1,
                    asset_2: 1}
    model.update(costs=0.1,
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = func(**{target_name: target})
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(getattr(portfolio, portfolio_target_name) - target) < threshold
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert portfolio.get_weight(asset_1) > portfolio_ref.get_weight(asset_1)
    assert portfolio.get_weight(asset_2) > portfolio_ref.get_weight(asset_2)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-3


def regularisation_testing(method_name: str,
                           target: float,
                           portfolio_target_name: 'str',
                           threshold: float):
    target_name = method_name.replace('mean', 'target')
    assets = get_assets()

    params = {target_name: target}

    coefs_params = [{'l1_coef': 0.1},
                    {'l2_coef': 0.1},
                    {'l1_coef': 0.1,
                     'l2_coef': 0.1}]

    # No short selling --> no impact of regularisation with l1 and impact with l2
    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    func = getattr(model, method_name)
    weights = func(**params)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets)

    for coef_param in coefs_params:
        coef_param.update(params)
        weights = func(**coef_param)
        portfolio = Portfolio(weights=weights,
                              assets=assets)
        assert abs(getattr(portfolio, portfolio_target_name) - target) < threshold
        assert abs(sum(portfolio.weights) - 1) < 1e-5
        if 'l1_coef' in params.keys():
            assert sum(abs(portfolio_ref.weights - portfolio.weights)) < 1e-4
        if 'l2_coef' in params.keys():
            assert sum(abs(portfolio_ref.weights - portfolio.weights)) > 1e-3
            assert sum(np.square(portfolio_ref.weights)) > sum(np.square(portfolio.weights))

    # Short Selling --> impact of regularisation
    model.update(weight_bounds=(None, None))
    weights = func(**params)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets)

    for coef_param in coefs_params:
        coef_param.update(params)
        weights = func(**coef_param)
        portfolio = Portfolio(weights=weights,
                              assets=assets)
        assert abs(getattr(portfolio, portfolio_target_name) - target) < threshold
        assert abs(sum(portfolio.weights) - 1) < 1e-5
        assert sum(abs(portfolio_ref.weights - portfolio.weights)) > 1e-3
        if 'l1_coef' in params.keys():
            assert sum(abs(portfolio_ref.weights)) > sum(abs(portfolio.weights))
        if 'l2_coef' in params.keys():
            assert sum(np.square(portfolio_ref.weights)) > sum(np.square(portfolio.weights))

    # Negative coef
    coefs_params = [{'l1_coef': -1},
                    {'l2_coef': -1}]
    for coef_param in coefs_params:
        try:
            coef_param.update(params)
            func(**coef_param)
            raise
        except ValueError:
            pass


def args_testing(method_name: str,
                 **kwargs):
    target_name = method_name.replace('mean', 'target')
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices = prices.iloc[:, :100].copy()

    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.99,
                         verbose=False)

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    func = getattr(model, method_name)

    # target is a float
    target = 0.05
    weights = func(**{target_name: target})
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (assets.asset_nb,)

    # target is a list or numpy array
    for target in [[0.02, 0.05], np.array([0.02, 0.05])]:
        weights = func(**{target_name: target})
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (len(target), assets.asset_nb)

    # Target is 0 or neg
    target = [1, -1]
    try:
        func(**{target_name: target})
        raise
    except ValueError:
        pass

    # Population
    population_size = 3
    weights = func(population_size=population_size)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (population_size, assets.asset_nb)

    # Both Population and Target is None
    try:
        func()
        raise
    except ValueError:
        pass

    # Both Population and Target is None
    try:
        func(**{target_name: target, 'population_size': 3})
        raise
    except ValueError:
        pass


def test_minimum():
    for param in PARAMS:
        print(param)
        minimum_testing(**param)


def test_population():
    for param in PARAMS:
        print(param)
        population_testing(**param)


def test_investment_type():
    for param in PARAMS:
        print(param)
        investment_type_testing(**param)


def test_costs():
    for param in PARAMS:
        print(param)
        costs_testing(**param)


def test_regularisation():
    for param in PARAMS:
        print(param)
        regularisation_testing(**param)


def test_args():
    for param in PARAMS:
        print(param)
        args_testing(**param)


def test_maximum_sharpe():
    assets = get_assets()

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, 0.1))

    weights = model.maximum_sharpe()
    portfolio = Portfolio(weights=weights,
                          assets=assets)

    portfolios_weights = model.mean_variance(population_size=30)
    # Remove nan
    portfolios_weights = portfolios_weights[~np.isnan(portfolios_weights).all(axis=1)]
    population = Population()
    for weights in portfolios_weights:
        population.append(Portfolio(weights=weights,
                                    assets=assets))

    max_sharpe_ptf = population.max(metric=Metrics.SHARPE_RATIO)

    assert abs(sum(portfolio.weights) - 1) < 1e-5
    assert abs(sum(max_sharpe_ptf.weights) - 1) < 1e-5
    assert abs(max_sharpe_ptf.sharpe_ratio - portfolio.sharpe_ratio) < 1e-2
    assert abs(max_sharpe_ptf.std - portfolio.std) < 1e-3
    assert abs(max_sharpe_ptf.mean - max_sharpe_ptf.mean) < 1e-4
