import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.optimization import *
from portfolio_optimization.loader import *
from portfolio_optimization.bloomberg import *


def test_inverse_volatility():
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    assets = load_assets(prices=prices, verbose=False)
    model = Optimization(assets=assets)
    weights = model.inverse_volatility()

    assert abs(sum(weights) - 1) < 1e-10
    w = 1 / assets.std
    w = w / sum(w)
    assert abs(weights - w).sum() < 1e-10


def test_equal_weighted():
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    assets = load_assets(prices=prices, verbose=False)
    model = Optimization(assets=assets)
    weights = model.equal_weighted()

    assert abs(sum(weights) - 1) < 1e-10
    w = 1 / assets.asset_nb
    assert abs(weights - w).sum() < 1e-10


def test_random():
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    assets = load_assets(prices=prices, verbose=False)
    model = Optimization(assets=assets)
    weights = model.random()

    assert abs(sum(weights) - 1) < 1e-10


def test_mean_variance():
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices = prices.iloc[:, :100].copy()

    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.99,
                         verbose=False)

    target_volatility = 0.02 / np.sqrt(255)

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Ref with no costs
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio_ref = Portfolio(weights=weights,
                              assets=assets,
                              name='ptf_ref')
    # uniform costs for all assets and empty prev_weight --> no impact on weights
    model.update(costs=0.1,
                 prev_w=None,
                 investment_duration_in_days=255)
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # uniform costs for all assets and uniform prev_weight --> no impact on weights
    model.update(prev_w=np.ones(assets.asset_nb))
    weights = model.mean_variance(target_volatility=target_volatility)
    portfolio = Portfolio(weights=weights,
                          assets=assets)
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
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert portfolio.get_weight(asset_1) > portfolio_ref.get_weight(asset_1)
    assert portfolio.get_weight(asset_2) > portfolio_ref.get_weight(asset_2)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-3

    # Population size
    portfolios_weights = model.mean_variance(population_size=30)
    assert portfolios_weights.shape[0] <= 30
    assert portfolios_weights.shape[1] == assets.asset_nb
