import numpy as np

from portfolio_optimization import *

if __name__ == '__main__':

    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    start_date = prices.index[0].date()
    end_date = prices.index[-1].date()
    target_annualized_volatility = 0.05
    target_variance = target_annualized_volatility**2 / 255
    train_duration = 900
    test_duration = 30

    population = Population()
    mpp = MultiPeriodPortfolio(name='mpp_test', tag='mpp_test')

    for train_period, test_period in walk_forward(start_date=start_date,
                                                  end_date=end_date,
                                                  train_duration=train_duration,
                                                  test_duration=test_duration):
        print(train_period)
        train, test = load_train_test_assets(prices=prices,
                                             train_period=train_period,
                                             test_period=test_period,
                                             pre_selection_number=50,
                                             pre_selection_correlation=0,
                                             verbose=False)
        model = Optimization(assets=train,
                                 investment_type=InvestmentType.FULLY_INVESTED,
                                 weight_bounds=(0, None))
        weights = model.mean_variance(target_variance=target_variance)
        if np.isnan(weights).any():
            print('no solution found')
            continue

        for tag, assets in [('train', train), ('test', test)]:
            portfolio = Portfolio(weights=weights,
                                  assets=assets,
                                  name=f'portfolio_{assets.name}',
                                  tag=tag)

            population.append(portfolio)
            if tag == 'test':
                mpp.append(portfolio)
    population.append(mpp)

    population.plot_metrics(x=Metrics.ANNUALIZED_STD, y=Metrics.ANNUALIZED_MEAN)
    print(population.composition(tags=['train']))
    population.plot_composition(tags=['train'])

    mpp.plot_returns()
    mpp.plot_cumulative_returns()
    mpp.plot_rolling_sharpe(days=20)
    print(mpp.sharpe_ratio)
    print(mpp.composition)
    mpp.plot_composition()
