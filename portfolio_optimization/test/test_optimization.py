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
    n = assets.asset_nb

    params = dict(expected_returns=assets.mu,
                  cov=assets.cov,
                  investment_type=InvestmentType.FULLY_INVESTED,
                  weight_bounds=(0, None),
                  target_variance=0.02 ** 2 / 255)

    # Ref with no costs
    portfolios_weights = mean_variance(**params)
    portfolio_ref = Portfolio(weights=portfolios_weights[0],
                              assets=assets,
                              name='ptf_ref')
    # Same costs for all assets and empty prev_weight --> no impact on weights
    portfolios_weights = mean_variance(costs=0.1,
                                       prev_w=None,
                                       **params)
    portfolio = Portfolio(weights=portfolios_weights[0],
                          assets=assets)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # Same costs for all assets and same prev_weight --> no impact on weights
    portfolios_weights = mean_variance(costs=0.1,
                                       prev_w=np.ones(n),
                                       **params)
    portfolio = Portfolio(weights=portfolios_weights[0],
                          assets=assets)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3

    # costs on one invested asset and same prev_weight --> impact on the invested asset weight
    portfolio_ref.composition
    costs = {'AGSBPKA FP Equity': 0.1,
             'C40 FP Equity': 0.2,
             'wrong_ticker': 0.4}
    self=assets

    portfolios_weights = mean_variance(costs=0.1,
                                       prev_w=np.ones(n),
                                       **params)
    portfolio = Portfolio(weights=portfolios_weights[0],
                          assets=assets)
    assert abs(portfolio.weights - portfolio_ref.weights).sum() < 1e-3


    portfolios_weights = mean_variance(expected_returns=assets.mu,
                                       cov=assets.cov,
                                       investment_type=InvestmentType.FULLY_INVESTED,
                                       weight_bounds=(0, None),
                                       target_variance=0.02 ** 2 / 255,
                                       costs=0,
                                       prev_w=None)

    portfolio_1 = Portfolio(weights=portfolios_weights[0],
                            assets=assets)

    print(portfolio_1.annualized_mean)
    print(portfolio_1.annualized_std)

    portfolios_weights = mean_variance(expected_returns=assets.mu,
                                       cov=assets.cov,
                                       investment_type=InvestmentType.FULLY_INVESTED,
                                       weight_bounds=(0, None),
                                       target_variance=0.02 ** 2 / 255,
                                       costs=1,
                                       prev_w=None)

    portfolio_2 = Portfolio(weights=portfolios_weights[0],
                            assets=assets)

    print(portfolio_2.annualized_mean)
    print(portfolio_2.annualized_std)
