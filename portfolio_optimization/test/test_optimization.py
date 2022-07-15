import datetime as dt
import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.optimization import *
from portfolio_optimization.loader import *
from portfolio_optimization.bloomberg import *

def test_mean_variance():
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    prices=prices.iloc[:, :100].copy()

    assets = load_assets(prices=prices,
                         asset_missing_threshold=0.1,
                         dates_missing_threshold=0.1,
                         removal_correlation=0.99,
                         verbose=True)

    # Efficient Frontier -- Mean Variance
    portfolios_weights = mean_variance(expected_returns=assets.mu,
                                       cov=assets.cov,
                                       investment_type=InvestmentType.FULLY_INVESTED,
                                       weight_bounds=(0, None),
                                       population_size=30)
    population=Population()
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 assets=assets,
                                 name=f'portfolio_{i}',
                                 tag='mean_variance'))

    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    hover_metrics=[Metrics.SHARPE_RATIO])
