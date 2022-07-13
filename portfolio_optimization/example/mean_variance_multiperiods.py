import plotly.express as px
import pandas as pd

from portfolio_optimization import *

if __name__ == '__main__':

    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    start_date = prices.index[int(2 * len(prices) / 3)].date()
    end_date = prices.index[-1].date()
    target_variance = 0.025 ** 2 / 255
    train_duration = 360
    test_duration = 30

    population = Population()
    mpp = MultiPeriodPortfolio(name='test')

    for train_period, test_period in walk_forward(start_date=start_date,
                                                  end_date=end_date,
                                                  train_duration=train_duration,
                                                  test_duration=test_duration):
        print(test_period)
        train, test = load_train_test_assets(prices=prices,
                                             train_period=train_period,
                                             test_period=test_period,
                                             pre_selection_correlation=-0.5,
                                             pre_selection_number=100,
                                             verbose=False)
        try:
            weights = mean_variance(expected_returns=train.mu,
                                    cov=train.cov,
                                    investment_type=InvestmentType.FULLY_INVESTED,
                                    weight_bounds=(0, None),
                                    target_variance=target_variance)
        except OptimizationError:
            continue

        for tag, assets in [('train', train), ('test', test)]:
            portfolio = Portfolio(weights=weights[0],
                                  assets=assets,
                                  name=assets.name,
                                  tag=tag)

            population.add(portfolio)
            if tag == 'test':
                mpp.add(portfolio)

    population.add(mpp)

    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN)
    population.composition()
    population.plot_composition()

    mpp.plot_returns()
    mpp.plot_prices_compounded()
    mpp.plot_rolling_sharpe(days=20)
    print(mpp.sharpe_ratio)
