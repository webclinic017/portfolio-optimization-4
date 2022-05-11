import datetime as dt
import numpy as np
import plotly.express as px
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.utils.tools import *
from portfolio_optimization.optimization.mean_variance import *
from portfolio_optimization.utils.assets import *
from portfolio_optimization.exception import *


def random_vs_mean_variance():
    """
    Compare the efficient frontier of the mean-variance optimization against portfolios of single asset and
    random portfolios
    """
    assets = load_assets(start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=200,
                         pre_selection_number=100)

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


def mean_variance_different_periods():
    """
    Compare the efficient frontier of the mean-variance optimization fitted on the train period (2018-2019) against the
    frontier tested on the test period (2019-2020)
    """
    assets_2018, assets_2019 = load_train_test_assets(train_period=(dt.date(2018, 1, 1), dt.date(2019, 1, 1)),
                                                      test_period=(dt.date(2019, 1, 1), dt.date(2020, 1, 1)),
                                                      random_selection=400,
                                                      pre_selection_number=100)

    population = Population()

    # Efficient Frontier -- Mean Variance -- Per period
    for assets in [assets_2018, assets_2019]:
        portfolios_weights = mean_variance(expected_returns=assets.mu,
                                           cov=assets.cov,
                                           investment_type=InvestmentType.FULLY_INVESTED,
                                           weight_bounds=(0, None),
                                           population_size=30)
        for i, weights in enumerate(portfolios_weights):
            population.add(Portfolio(weights=weights,
                                     fitness_type=FitnessType.MEAN_STD,
                                     assets=assets,
                                     pid=f'train_{assets.name}_{i}',
                                     name=str(i),
                                     tag=f'train_{assets.name}'))

    # Test the portfolios on the test period
    for portfolio in population.get_portfolios(tags=f'train_{assets_2018.name}'):
        population.add(Portfolio(weights=portfolio.weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets_2019,
                                 pid=f'test_{assets.name}_{portfolio.name}',
                                 name=portfolio.name,
                                 tag=f'test_{assets_2019.name}'))

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SHARPE_RATIO)
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale='name',
                    tags=[f'train_{assets_2018.name}', f'test_{assets_2019.name}'])
    population.plot_composition(tags=[f'train_{assets_2018.name}', f'train_{assets_2019.name}'])

    # Metrics
    max_sortino = population.max(metric=Metrics.SORTINO_RATIO)
    print(max_sortino.sortino_ratio)


def mean_variance_out_of_sample():
    """

    """
    population = Population()
    metrics = []
    target_variance = 0.025 ** 2 / 255

    start = dt.date(2018, 1, 1)
    train_duration = 150
    test_duration = 150
    rolling_period = 30
    period = -1

    while True:
        period += 1
        train_start = start + dt.timedelta(days=period * rolling_period)
        train_end = train_start + dt.timedelta(days=train_duration)
        test_start = train_end
        test_end = test_start + dt.timedelta(days=test_duration)

        train, test = load_train_test_assets(train_period=(train_start, train_end),
                                             test_period=(test_start, test_end),
                                             correlation_threshold_pre_selection=-0.5,
                                             pre_selection_number=100)

        if test.date_nb < test_duration / 2:
            break
        try:
            portfolios_weights = mean_variance(expected_returns=train.mu,
                                               cov=train.cov,
                                               investment_type=InvestmentType.FULLY_INVESTED,
                                               weight_bounds=(0, None),
                                               target_variance=target_variance)
        except OptimizationError:
            continue

        train_portfolio = Portfolio(weights=portfolios_weights[0],
                                    fitness_type=FitnessType.MEAN_STD,
                                    assets=train,
                                    pid=f'train_{train.name}',
                                    name=f'train_{train.name}',
                                    tag=f'train')

        test_portfolio = Portfolio(weights=portfolios_weights[0],
                                   fitness_type=FitnessType.MEAN_STD,
                                   assets=test,
                                   pid=f'test_{test.name}',
                                   name=f'test_{test.name}',
                                   tag=f'test')

        population.add(train_portfolio)
        population.add(test_portfolio)

        metrics.append({'train_sharpe': train_portfolio.sharpe_ratio,
                        'train_sric': train_portfolio.sric,
                        'test_sharpe': test_portfolio.sharpe_ratio})

    df = pd.DataFrame(metrics)
    fig = px.scatter(df)
    fig.show()

    print(df.mean())

    """
    train_sharpe    6.154613
    train_sric      6.078020
    test_sharpe     2.088287
    
    train_sharpe    5.678279
    train_sric      5.590826
    test_sharpe     1.275128
    """

