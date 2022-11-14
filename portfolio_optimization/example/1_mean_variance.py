import datetime as dt
import numpy as np

from portfolio_optimization import (EXAMPLE_PRICES_PATH, InvestmentType, Metrics, Portfolio,
                                    Population, Optimization, load_assets, load_prices)

if __name__ == '__main__':

    """
    Compare the efficient frontier of the mean-variance optimization against random, equally weighted,
    inverse vol and single asset portfolios.
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         pre_selection_number=100,
                         pre_selection_correlation=0)

    population = Population()

    # Portfolios of one asset
    for i in range(assets.asset_nb):
        weights = np.zeros(assets.asset_nb)
        weights[i] = 1
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'single_asset_{i}',
                                    tag='single_asset'))

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    model.minimum_variance2()
    model.mean_variance2()
    model.minimum_variance()

    # Random portfolios
    for i in range(10):
        population.append(Portfolio(weights=model.random(),
                                    assets=assets,
                                    name=f'random_{i}',
                                    tag='random'))

    # Inverse Volatility
    population.append(Portfolio(weights=model.inverse_volatility(),
                                assets=assets,
                                name='inverse_volatility'))

    # Equal Weighted
    population.append(Portfolio(weights=model.equal_weighted(),
                                assets=assets,
                                name='equal_weighted'))

    # Efficient Frontier -- Mean Variance
    portfolios_weights = model.mean_variance(population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_variance_{i}',
                                    tag='mean_variance'))

    # Plot
    population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SHARPE_RATIO,
                            hover_metrics=[Metrics.MAX_DRAWDOWN, Metrics.SORTINO_RATIO])

    # Find the portfolio with maximum Sharpe Ratio
    max_sharpe_ptf = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe_ptf.sharpe_ratio)
    print(max_sharpe_ptf.summary())

    # Find the portfolio with maximum CDaR 95% Ratio
    max_cdar_95_ptf = population.max(metric=Metrics.CDAR_95_RATIO)
    print(max_cdar_95_ptf.cdar_95_ratio)
    print(max_cdar_95_ptf.summary())

    # Combination of the two portfolio
    ptf = max_sharpe_ptf * 0.5 + max_cdar_95_ptf * 0.5
    ptf.name = 'sharpe_cdar'
    population.append(ptf)

    names = [max_sharpe_ptf.name, max_cdar_95_ptf.name, ptf.name, 'equal_weighted', 'inverse_volatility']

    # Display summaries:
    population.summary(names=names)

    # Plot compositions
    population.plot_composition(names=names)

    # Plot cumulative returns
    population.plot_cumulative_returns(names=names)
