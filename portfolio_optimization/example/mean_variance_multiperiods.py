import datetime as dt
import plotly.express as px
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization.mean_variance import *
from portfolio_optimization.utils.assets import *
from portfolio_optimization.exception import *
from portfolio_optimization.bloomberg.loader import *

if __name__ == '__main__':

    """

    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    population = Population()
    metrics = []
    target_variance = 0.025 ** 2 / 255

    start = dt.date(2018, 1, 1)
    train_duration = 150
    test_duration = 150
    rolling_period = 30
    period = -1

    multi_period_portfolio_train = MultiPeriodPortfolio(name='train')
    multi_period_portfolio_test = MultiPeriodPortfolio(name='test')

    while True:
        period += 1
        train_start = start + dt.timedelta(days=period * rolling_period)
        train_end = train_start + dt.timedelta(days=train_duration)
        test_start = train_end
        test_end = test_start + dt.timedelta(days=test_duration)

        train, test = load_train_test_assets(prices=prices,
                                             train_period=(train_start, train_end),
                                             test_period=(test_start, test_end),
                                             correlation_threshold_pre_selection=-0.5,
                                             pre_selection_number=100)

        if test.date_nb < test_duration / 2:
            break

        print(train)
        try:
            weights = mean_variance(expected_returns=train.mu,
                                    cov=train.cov,
                                    investment_type=InvestmentType.FULLY_INVESTED,
                                    weight_bounds=(0, None),
                                    target_variance=target_variance)
        except OptimizationError:
            continue

        for tag, asset in [('train', train), ('test', test)]:
            multi_period_portfolio_train.add(Portfolio(weights=weights[0],
                                                       assets=asset,
                                                       pid=asset.name,
                                                       name=asset.name,
                                                       tag=tag))

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
