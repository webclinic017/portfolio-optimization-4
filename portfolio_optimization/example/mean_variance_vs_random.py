import datetime as dt
import numpy as np

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.optimization.variance import *
from portfolio_optimization.utils.assets import *
from portfolio_optimization.bloomberg.loader import *

if __name__ == '__main__':

    """
    Compare the efficient frontier of the mean-variance optimization against portfolios of single asset and
    random portfolios
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=200,
                         pre_selection_number=100,
                         pre_selection_correlation=0)

    population = Population()
    # Portfolios of one asset
    for i in range(assets.asset_nb):
        weights = np.zeros(assets.asset_nb)
        weights[i] = 1
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 tag='single_asset'))

    # Random portfolios
    for _ in range(10):
        weights = rand_weights_dirichlet(n=assets.asset_nb)
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 tag='random'))

    # Efficient Frontier -- Mean Variance
    portfolios_weights = mean_variance(expected_returns=assets.mu,
                                       cov=assets.cov,
                                       investment_type=InvestmentType.FULLY_INVESTED,
                                       weight_bounds=(0, None),
                                       population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 name=str(i),
                                 tag='mean_variance'))

    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN, color_scale=Metrics.SHARPE_RATIO)
