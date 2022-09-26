import datetime as dt
import numpy as np

from portfolio_optimization import *

if __name__ == '__main__':

    """
    Compare the efficient frontier of the mean-variance optimization against random, equally weighted,
    inverse vol and single asset portfolios.
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
                                 assets=assets,
                                 name=f'single_asset_{i}',
                                 tag='single_asset'))

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Random portfolios
    for i in range(10):
        weights = model.random()
        population.add(Portfolio(weights=weights,
                                 assets=assets,
                                 name=f'random_{i}',
                                 tag='random'))

    # Inverse Volatility
    weights = model.inverse_volatility()
    population.add(Portfolio(weights=weights,
                             assets=assets,
                             name='inverse_volatility'))

    # Equal Weighted
    weights = model.equal_weighted()
    population.add(Portfolio(weights=weights,
                             assets=assets,
                             name='equal_weighted'))

    # Efficient Frontier -- Mean Variance
    portfolios_weights = model.mean_variance(population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 assets=assets,
                                 name=f'mean_variance_{i}',
                                 tag='mean_variance'))

    population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SHARPE_RATIO,
                            hover_metrics=[Metrics.MAX_DRAWDOWN, Metrics.SORTINO_RATIO])

    # Metrics
    max_sharpe_ptf = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe_ptf.sharpe_ratio)
    print(max_sharpe_ptf.summary())

    max_cdar_95_ptf = population.max(metric=Metrics.CDAR_95_RATIO)
    print(max_cdar_95_ptf.sharpe_ratio)
    print(max_cdar_95_ptf.summary())

    # Composition
    population.plot_composition(names=[max_sharpe_ptf.name,
                                       max_cdar_95_ptf.name,
                                       'equal_weighted',
                                       'inverse_volatility'])

    # Prices
    population.plot_cumulative_returns(names=[max_sharpe_ptf.name,
                                              max_cdar_95_ptf.name,
                                              'equal_weighted',
                                              'inverse_volatility'])
