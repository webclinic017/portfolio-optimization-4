import datetime as dt

from portfolio_optimization import *

if __name__ == '__main__':
    """
    Compare the efficient frontier of the mean-variance optimization fitted on the train period (2018-2019) against the
    frontier tested on the test period (2019-2020)
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    assets_train, assets_test = load_train_test_assets(prices=prices,
                                                       train_period=(dt.date(2018, 1, 1), dt.date(2019, 1, 1)),
                                                       test_period=(dt.date(2019, 1, 1), dt.date(2020, 1, 1)),
                                                       pre_selection_number=50,
                                                       pre_selection_correlation=0)

    population = Population()

    # Efficient Frontier -- Mean Variance -- Per period
    for assets in [assets_train, assets_test]:
        model = Optimization(assets=assets,
                             investment_type=InvestmentType.FULLY_INVESTED,
                             weight_bounds=(0, None))
        portfolios_weights = model.mean_variance(population_size=30)
        for i, weights in enumerate(portfolios_weights):
            population.append(Portfolio(weights=weights,
                                        assets=assets,
                                        name=f'train_{assets.name}_{i}',
                                        tag=f'train_{assets.name}'))

    # Test the portfolios on the test period
    for portfolio in population.get_portfolios(tags=f'train_{assets_train.name}'):
        population.append(Portfolio(weights=portfolio.weights,
                                    assets=assets_test,
                                    name=f'test_{assets_test.name}_{portfolio.name.split("_")[-1]}',
                                    tag=f'test_{assets_test.name}'))

    # Plot
    population.plot_metrics(x=Metric.ANNUALIZED_STD,
                            y=Metric.ANNUALIZED_MEAN,
                            color_scale=Metric.SHARPE_RATIO)
    population.plot_metrics(x=Metric.ANNUALIZED_STD,
                            y=Metric.ANNUALIZED_MEAN,
                            color_scale='tag',
                            names=[f'{x}_{assets_test.name}_{i}' for i in range(10) for x in ['train', 'test']]+
                                  [f'train_{assets_train.name}_{i}' for i in range(10) ])
    population.plot_composition(tags=[f'train_{assets_train.name}', f'train_{assets_test.name}'])

    # Metrics
    max_sortino = population.max(metric=Metric.SORTINO_RATIO)
    print(max_sortino.sortino_ratio)

