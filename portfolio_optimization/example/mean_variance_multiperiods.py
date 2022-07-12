import plotly.express as px
import pandas as pd

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization.variance import *
from portfolio_optimization.utils.assets import *
from portfolio_optimization.exception import *
from portfolio_optimization.bloomberg.loader import *
from portfolio_optimization.utils.tools import *

if __name__ == '__main__':

    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    start_date = prices.index[0].date()
    end_date = prices.index[-1].date()
    target_variance = 0.025 ** 2 / 255

    population = Population()
    test_multi_period_portfolio = MultiPeriodPortfolio(name='test')

    for train_period, test_period in walk_forward(start_date=start_date,
                                                  end_date=end_date,
                                                  train_duration=360,
                                                  test_duration=90):
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
                                  pid=assets.name,
                                  name=assets.name,
                                  tag=tag)

            population.add(portfolio)
            if tag == 'test':
                test_multi_period_portfolio.add(portfolio)


    population.plot(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN)
    population.plot_composition()


    test_multi_period_portfolio.plot_rolling_sharpe(days=20)
    test_multi_period_portfolio.plot_rolling_sharpe(days=20)


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
