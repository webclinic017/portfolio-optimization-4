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
    prices=prices.iloc[:, :20].copy()

    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         verbose=True)

    # Efficient Frontier -- Mean Variance
    portfolios_weights = mean_variance(expected_returns=assets.mu,
                                       cov=assets.cov,
                                       investment_type=InvestmentType.FULLY_INVESTED,
                                       weight_bounds=(0, None),
                                       target_variance=0.02**2/360)

