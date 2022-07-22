import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization import *
from portfolio_optimization.loader import *
from portfolio_optimization.bloomberg import *
from portfolio_optimization.assets import *


def get_assets() -> Assets:
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices = prices.iloc[:, :100].copy()
    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.99,
                         verbose=False)

    return assets


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


def test_mean_variance():
    assets = get_assets()

    target_volatility = 0.02 / np.sqrt(255)

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Ref with no costs
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')
    assert abs(portfolio_ref.std - target_volatility) < 1e-8

    # uniform costs for all assets and empty prev_weight --> no impact on weights
    model.update(costs=0.1,
                 prev_w=None,
                 investment_duration_in_days=255)
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.std - target_volatility) < 1e-8
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # uniform costs for all assets and uniform prev_weight --> no impact on weights
    model.update(prev_w=np.ones(assets.asset_nb))
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.std - target_volatility) < 1e-8
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # costs on top two invested assets and uniform prev_weight --> impact on the two invested assets weight
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.1}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=None)
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.std - target_volatility) < 1e-8
    assert asset_1 not in portfolio.composition.index
    assert asset_2 not in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-3

    # costs and identical prev_weight on top two invested assets --> the top two assets weights stay > 0
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.1}
    prev_weights = {asset_1: portfolio_ref.get_weight(asset_name=asset_1),
                    asset_2: portfolio_ref.get_weight(asset_name=asset_2)}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.std - target_volatility) < 1e-8
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # identical costs on all assets and large prev_weight on top two invested assets
    # --> the top two assets weights become larger
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    prev_weights = {asset_1: 1,
                    asset_2: 1}
    model.update(costs=0.1,
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.std - target_volatility) < 1e-8
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert portfolio.get_weight(asset_1) > portfolio_ref.get_weight(asset_1)
    assert portfolio.get_weight(asset_2) > portfolio_ref.get_weight(asset_2)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-3

    # Population size
    portfolios_weights = model.mean_variance(population_size=30, ignore_none=False)
    assert len(portfolios_weights) == 30
    assert portfolios_weights[0].shape == (assets.asset_nb,)


def test_mean_variance_regularisation():
    assets = get_assets()

    target_volatility = 0.02 / np.sqrt(255)

    # No short selling --> no impact of regularisation
    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')

    # L1 coef
    weights = model.mean_variance(target_volatility=target_volatility,
                                  l1_coef=1)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.std - target_volatility) < 1e-8
    assert abs(sum(portfolio.weights) - 1) < 1e-10
    assert sum(abs(portfolio_ref.weights - portfolio.weights)) < 1e-4
    try:
        model.mean_variance(target_volatility=target_volatility,
                            l1_coef=-1)
        raise
    except ValueError:
        pass

    # Short Selling
    model.update(weight_bounds=(None, None))
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')

    # L1 coef
    weights = model.mean_variance(target_volatility=target_volatility,
                                  l1_coef=1)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.std - target_volatility) < 1e-8
    assert abs(sum(portfolio.weights) - 1) < 1e-10
    assert len(portfolio.composition) < len(portfolio_ref.composition)


def test_maximum_sharpe():
    assets = get_assets()

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, 0.1))

    weights = model.maximum_sharpe()
    portfolio = Portfolio(weights=weights,
                          assets=assets)

    portfolios_weights = model.mean_variance(population_size=30)
    population = Population()
    for weights in portfolios_weights:
        population.add(Portfolio(weights=weights,
                                 assets=assets))

    max_sharpe_ptf = population.max(metric=Metrics.SHARPE_RATIO)

    assert abs(sum(portfolio.weights) - 1) < 1e-5
    assert abs(sum(max_sharpe_ptf.weights) - 1) < 1e-5
    assert abs(max_sharpe_ptf.sharpe_ratio - portfolio.sharpe_ratio) < 1e-2
    assert abs(max_sharpe_ptf.std - portfolio.std) < 1e-3
    assert abs(max_sharpe_ptf.mean - max_sharpe_ptf.mean) < 1e-4


def test_mean_semivariance():
    assets = get_assets()

    target_semideviation = 0.02 / np.sqrt(255)

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Ref with no costs
    weights = model.mean_semivariance(target_semideviation=target_semideviation)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')
    assert abs(portfolio_ref.downside_std - target_semideviation) < 1e-4
    # uniform costs for all assets and empty prev_weight --> no impact on weights
    model.update(costs=0.1,
                 prev_w=None,
                 investment_duration_in_days=255)
    weights = model.mean_semivariance(target_semideviation=target_semideviation)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.downside_std - target_semideviation) < 1e-4
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # uniform costs for all assets and uniform prev_weight --> no impact on weights
    model.update(prev_w=np.ones(assets.asset_nb))
    weights = model.mean_semivariance(target_semideviation=target_semideviation)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.downside_std - target_semideviation) < 1e-4
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # costs on top two invested assets and uniform prev_weight --> impact on the two invested assets weight
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.1}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=None)
    weights = model.mean_semivariance(target_semideviation=target_semideviation)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.downside_std - target_semideviation) < 1e-4
    assert asset_1 not in portfolio.composition.index
    assert asset_2 not in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-1

    # costs and identical prev_weight on top two invested assets --> the top two assets weights stay > 0
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.1}
    prev_weights = {asset_1: portfolio_ref.get_weight(asset_name=asset_1),
                    asset_2: portfolio_ref.get_weight(asset_name=asset_2)}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_semivariance(target_semideviation=target_semideviation)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.downside_std - target_semideviation) < 1e-4
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # identical costs on all assets and large prev_weight on top two invested assets
    # --> the top two assets weights become larger
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    prev_weights = {asset_1: 1,
                    asset_2: 1}
    model.update(costs=0.1,
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_semivariance(target_semideviation=target_semideviation)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.downside_std - target_semideviation) < 1e-4
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert portfolio.get_weight(asset_1) > portfolio_ref.get_weight(asset_1)
    assert portfolio.get_weight(asset_2) > portfolio_ref.get_weight(asset_2)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-1

    # Population size
    portfolios_weights = model.mean_semivariance(population_size=30)
    assert portfolios_weights.shape[0] <= 30
    assert portfolios_weights.shape[1] == assets.asset_nb


def test_mean_cvar():
    assets = get_assets()

    target_cvar = 0.01

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Ref with no costs
    weights = model.mean_cvar(target_cvar=target_cvar)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')
    assert abs(portfolio_ref.cvar_95 - target_cvar) < 1e-4
    # uniform costs for all assets and empty prev_weight --> no impact on weights
    model.update(costs=0.1,
                 prev_w=None,
                 investment_duration_in_days=255)
    weights = model.mean_cvar(target_cvar=target_cvar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.cvar_95 - target_cvar) < 1e-4
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # uniform costs for all assets and uniform prev_weight --> no impact on weights
    model.update(prev_w=np.ones(assets.asset_nb))
    weights = model.mean_cvar(target_cvar=target_cvar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.cvar_95 - target_cvar) < 1e-4
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # costs on top two invested assets and uniform prev_weight --> impact on the two invested assets weight
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.1}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=None)
    weights = model.mean_cvar(target_cvar=target_cvar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.cvar_95 - target_cvar) < 1e-4
    assert asset_1 not in portfolio.composition.index
    assert asset_2 not in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-1

    # costs and identical prev_weight on top two invested assets --> the top two assets weights stay > 0
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.1}
    prev_weights = {asset_1: portfolio_ref.get_weight(asset_name=asset_1),
                    asset_2: portfolio_ref.get_weight(asset_name=asset_2)}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_cvar(target_cvar=target_cvar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.cvar_95 - target_cvar) < 1e-4
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # identical costs on all assets and large prev_weight on top two invested assets
    # --> the top two assets weights become larger
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    prev_weights = {asset_1: 1,
                    asset_2: 1}
    model.update(costs=0.1,
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_cvar(target_cvar=target_cvar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.cvar_95 - target_cvar) < 1e-4
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert portfolio.get_weight(asset_1) > portfolio_ref.get_weight(asset_1)
    assert portfolio.get_weight(asset_2) > portfolio_ref.get_weight(asset_2)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-1

    # Population size
    portfolios_weights = model.mean_cvar(population_size=30)
    assert portfolios_weights.shape[0] <= 30
    assert portfolios_weights.shape[1] == assets.asset_nb


def test_mean_cdar():
    assets = get_assets()

    target_cdar = 0.05

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Ref with no costs
    weights = model.mean_cdar(target_cdar=target_cdar)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')
    assert abs(portfolio_ref.cdar_95 - target_cdar) < 5e-3
    # uniform costs for all assets and empty prev_weight --> no impact on weights
    model.update(costs=0.1,
                 prev_w=None,
                 investment_duration_in_days=255)
    weights = model.mean_cdar(target_cdar=target_cdar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio_ref.cdar_95 - target_cdar) < 5e-3
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # uniform costs for all assets and uniform prev_weight --> no impact on weights
    model.update(prev_w=np.ones(assets.asset_nb))
    weights = model.mean_cdar(target_cdar=target_cdar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio_ref.cdar_95 - target_cdar) < 5e-3
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # costs on top two invested assets and uniform prev_weight --> impact on the two invested assets weight
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.5}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=None)
    weights = model.mean_cdar(target_cdar=target_cdar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio_ref.cdar_95 - target_cdar) < 5e-3
    assert asset_1 not in portfolio.composition.index
    assert asset_2 not in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-1

    # costs and identical prev_weight on top two invested assets --> the top two assets weights stay > 0
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.5}
    prev_weights = {asset_1: portfolio_ref.get_weight(asset_name=asset_1),
                    asset_2: portfolio_ref.get_weight(asset_name=asset_2)}
    model.update(costs=assets.dict_to_array(assets_dict=costs),
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_cdar(target_cdar=target_cdar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio_ref.cdar_95 - target_cdar) < 5e-3
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-1

    # identical costs on all assets and large prev_weight on top two invested assets
    # --> the top two assets weights become larger
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    prev_weights = {asset_1: 1,
                    asset_2: 1}
    model.update(costs=0.1,
                 prev_w=assets.dict_to_array(assets_dict=prev_weights))
    weights = model.mean_cdar(target_cdar=target_cdar)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio_ref.cdar_95 - target_cdar) < 5e-3
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert portfolio.get_weight(asset_1) > portfolio_ref.get_weight(asset_1)
    assert portfolio.get_weight(asset_2) > portfolio_ref.get_weight(asset_2)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-1

    # Population size
    portfolios_weights = model.mean_cdar(population_size=30)
    assert portfolios_weights.shape[0] <= 30
    assert portfolios_weights.shape[1] == assets.asset_nb


def test_optimization_args():
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

    args_name = {
        'mean_variance': 'target_volatility',
        'mean_semivariance': 'target_semideviation',
        'mean_cvar': 'target_cvar',
        'mean_cdar': 'target_cdar',
    }

    for method_name, arg_name in args_name.items():
        func = getattr(model, method_name)

        # target is a float
        target = 0.05
        weights = func(**{arg_name: target})
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (assets.asset_nb,)

        # target is a list or numpy array
        for target in [[0.02, 0.05], np.array([0.02, 0.05])]:
            weights = func(**{arg_name: target, 'ignore_none': False})
            assert isinstance(weights, list)
            assert len(weights) == len(target)
            for w in weights:
                assert isinstance(w, np.ndarray)
                assert w.shape == (assets.asset_nb,)

        # Ignore None
        target = [1e10]
        weights = func(**{arg_name: target, 'ignore_none': False})
        assert isinstance(weights, list)
        assert len(weights) == len(target)
        assert weights[0] is None
        weights = func(**{arg_name: target, 'ignore_none': True})
        assert isinstance(weights, list)
        assert len(weights) < len(target)

        # Target is 0 or neg
        target = [1, 0]
        try:
            func(**{arg_name: target})
            raise
        except ValueError:
            pass

        # Population
        population_size = 3
        weights = func(population_size=population_size, ignore_none=False)
        assert isinstance(weights, list)
        assert len(weights) == population_size
        for w in weights:
            assert isinstance(w, np.ndarray)
            assert w.shape == (assets.asset_nb,)

        # Both Population and Target is None
        try:
            func()
            raise
        except ValueError:
            pass

        # Both Population and Target is None
        try:
            func(**{arg_name: target, 'population_size': 3})
            raise
        except ValueError:
            pass
