import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.optimization import *
from portfolio_optimization.loader import *
from portfolio_optimization.bloomberg import *


def test_mean_variance():
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices = prices.iloc[:, :100].copy()

    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.99,
                         verbose=False)

    target_variance=0.02 ** 2 / 255

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Ref with no costs
    portfolios_weights = model.mean_variance(target_variance=target_variance)
    portfolio_ref = Portfolio(weights=portfolios_weights[0],
                              assets=assets,
                              name='ptf_ref')
    # uniform costs for all assets and empty prev_weight --> no impact on weights
    model.update(costs=0.1,
                 prev_w=None,
                 investment_duration_in_days=255)
    portfolios_weights = model.mean_variance(target_variance=target_variance)
    portfolio = Portfolio(weights=portfolios_weights[0],
                          assets=assets)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # uniform costs for all assets and uniform prev_weight --> no impact on weights
    portfolios_weights = mean_variance(costs=0.1,
                                       prev_w=np.ones(n),
                                       investment_duration_in_days=255,
                                       **params)
    portfolio = Portfolio(weights=portfolios_weights[0],
                          assets=assets)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # costs on top two invested assets and uniform prev_weight --> impact on the two invested assets weight
    asset_1 = portfolio_ref.composition.index[0]
    asset_2 = portfolio_ref.composition.index[1]
    costs = {asset_1: 0.2,
             asset_2: 0.1}
    portfolios_weights = mean_variance(costs=assets.dict_to_array(assets_dict=costs),
                                       prev_w=None,
                                       investment_duration_in_days=255,
                                       **params)
    portfolio = Portfolio(weights=portfolios_weights[0],
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

    portfolios_weights = mean_variance(costs=assets.dict_to_array(assets_dict=costs),
                                       prev_w=assets.dict_to_array(assets_dict=prev_weights),
                                       investment_duration_in_days=255,
                                       **params)
    portfolio = Portfolio(weights=portfolios_weights[0],
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

    portfolios_weights = mean_variance(costs=0.1,
                                       prev_w=assets.dict_to_array(assets_dict=prev_weights),
                                       investment_duration_in_days=255,
                                       **params)
    portfolio = Portfolio(weights=portfolios_weights[0],
                          assets=assets)
    assert asset_1 in portfolio.composition.index
    assert asset_2 in portfolio.composition.index
    assert portfolio.get_weight(asset_1) > portfolio_ref.get_weight(asset_1)
    assert portfolio.get_weight(asset_2) > portfolio_ref.get_weight(asset_2)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() > 1e-3
