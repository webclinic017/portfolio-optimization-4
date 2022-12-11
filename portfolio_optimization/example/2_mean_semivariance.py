import datetime as dt

from portfolio_optimization import *
from portfolio_optimization.utils.tools import *

if __name__ == '__main__':
    """
    Compare the Efficient Frontier of the mean-variance against the mean-semivariance optimization
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=200,
                         pre_selection_number=50,
                         pre_selection_correlation=0)

    population = Population()

    model = Optimization(assets=assets,
                         weight_bounds=(0, None))

    # Efficient Frontier -- Mean Variance
    portfolios_weights = model.mean_variance(population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_variance_{i}',
                                    tag='mean_variance'))

    # Efficient Frontier -- Mean Semivariance
    portfolios_weights = model.mean_semi_variance(population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_semivariance_{i}',
                                    tag='mean_semivariance'))

    # Plot
    population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SHARPE_RATIO,
                            hover_metrics=[Metrics.SORTINO_RATIO])
    population.plot_metrics(x=Metrics.ANNUALIZED_SEMISTD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SORTINO_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])

    # Metrics
    max_sharpe_ptf = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe_ptf.sharpe_ratio)
    print(max_sharpe_ptf.summary())

    max_sortino_ptf = population.max(metric=Metrics.SORTINO_RATIO)
    print(max_sortino_ptf.sortino_ratio)
    print(max_sharpe_ptf.summary())

    # Composition
    population.plot_composition(names=[max_sharpe_ptf.name, max_sortino_ptf.name])

    # Prices
    population.plot_cumulative_returns(names=[max_sharpe_ptf.name, max_sortino_ptf.name])

