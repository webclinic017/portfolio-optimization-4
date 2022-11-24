import datetime as dt

from portfolio_optimization import (EXAMPLE_PRICES_PATH, InvestmentType, Metrics, Portfolio,
                                    Population, Optimization, load_assets, load_prices)

if __name__ == '__main__':
    """
    Compare the Efficient Frontier of the mean-variance against the mean-cdar optimization
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)
    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         pre_selection_number=100,
                         pre_selection_correlation=0)
    population = Population()

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    # Efficient Frontier -- Mean Variance
    portfolios_weights = model.mean_variance(population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_variance_{i}',
                                    tag='mean_variance'))

    # Efficient Frontier -- Mean CDaR
    portfolios_weights = model.mean_cdar(beta=0.95,
                                         population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_cdar_{i}',
                                    tag='mean_cdar'))

    # Plot
    population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SHARPE_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])
    population.plot_metrics(x=Metrics.ANNUALIZED_DOWNSIDE_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SORTINO_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])
    population.plot_metrics(x=Metrics.MAX_DRAWDOWN,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CALMAR_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])
    population.plot_metrics(x=Metrics.CDAR,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CDAR_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])

    # Metrics
    max_sharpe = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe.sharpe_ratio)

    max_cdar_95_ratio = population.max(metric=Metrics.CDAR_RATIO)
    print(max_cdar_95_ratio.cdar_ratio)

    # Composition
    population.plot_composition(names=[max_sharpe.name, max_cdar_95_ratio.name])

    # Prices
    population.plot_cumulative_returns(names=[max_sharpe.name, max_cdar_95_ratio.name])


def mean_cdar_vs_mean_cvar():
    """
    Compare the Efficient Frontier of the mean-cdar against the mean-cvar optimization
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=200,
                         pre_selection_number=100,
                         pre_selection_correlation=0)

    population = Population()

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))

    # Efficient Frontier -- Mean CDaR
    portfolios_weights = model.mean_cdar(beta=0.95,
                                         population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_cdar_{i}',
                                    tag='mean_cdar'))

    # Efficient Frontier -- Mean CVaR
    portfolios_weights = model.mean_cvar(beta=0.95,
                                         population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_cvar_{i}',
                                    tag='mean_cvar'))

    # Plot
    population.plot_metrics(x=Metrics.CDAR,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CDAR_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])
    population.plot_metrics(x=Metrics.CVAR,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CVAR_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])

    # Metrics
    max_cdar_95_ratio = population.max(metric=Metrics.CDAR_RATIO)
    print(max_cdar_95_ratio.cdar_ratio)

    max_cvar_95_ratio = population.max(metric=Metrics.CVAR_RATIO)
    print(max_cvar_95_ratio.cvar_ratio)

    # Composition
    population.plot_composition(names=[max_cdar_95_ratio.name, max_cvar_95_ratio.name])

    # Prices
    population.plot_cumulative_returns(names=[max_cdar_95_ratio.name, max_cvar_95_ratio.name])
