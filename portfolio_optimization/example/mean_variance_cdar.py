import datetime as dt
import numpy as np
from portfolio_optimization.meta import InvestmentType, Metrics
from portfolio_optimization.paths import EXAMPLE_PRICES_PATH
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.population import Population
from portfolio_optimization.optimization import Optimization
from portfolio_optimization.loader import load_assets
from portfolio_optimization.bloomberg.loader import load_prices


def mean_variance_cdar():
    """
    Compare the Efficient Frontier of the mean-variance against the mean-cdar optimization
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=200,
                         pre_selection_number=100,
                         pre_selection_correlation=0)

    population = Population()

    target_volatility = 0.03
    target_cdar = 0.04

    model = Optimization(assets=assets,
                         investment_type=InvestmentType.FULLY_INVESTED,
                         weight_bounds=(0, None))
    # Efficient Frontier -- Mean Variance
    portfolios_weights = model.mean_variance(population_size=10)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_variance_{i}',
                                    tag='mean_variance'))

    # Efficient Frontier -- Mean CDaR
    portfolios_weights = model.mean_cdar(beta=0.95,
                                         population_size=10)

    for i, weights in enumerate([portfolios_weights]):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_cdar_{i}',
                                    tag='mean_cdar'))

    # Efficient Frontier -- Mean Variance CDaR
    portfolios_weights = model.mean_variance_cdar(beta=0.95,
                                                  population_size=100)
    portfolios_weights = portfolios_weights.reshape(-1,assets.asset_nb)
    for i, weights in enumerate(portfolios_weights):
        if not np.isnan(weights).all():
            population.append(Portfolio(weights=weights,
                                        assets=assets,
                                        name=f'mean_variance_cdar_{i}',
                                        tag='mean_variance_cdar'))

    # Plot
    population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SHARPE_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])
    population.plot_metrics(x=Metrics.CDAR_95,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CDAR_95_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])

    population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                            y=Metrics.CDAR_95,
                            z=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CDAR_95_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])


    ptf1 = population.get_portfolios(tags='mean_variance')[0]
    ptf2 = population.get_portfolios(tags='mean_cdar')[0]
    ptf3 = population.get_portfolios(tags='mean_variance_cdar')[0]

    # Metrics
    max_sharpe = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe.sharpe_ratio)

    max_cdar_95_ratio = population.max(metric=Metrics.CDAR_95_RATIO)
    print(max_cdar_95_ratio.cdar_95_ratio)

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
    population.plot_metrics(x=Metrics.CDAR_95,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CDAR_95_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])
    population.plot_metrics(x=Metrics.CVAR_95,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CVAR_95_RATIO,
                            hover_metrics=[Metrics.SHARPE_RATIO])

    # Metrics
    max_cdar_95_ratio = population.max(metric=Metrics.CDAR_95_RATIO)
    print(max_cdar_95_ratio.cdar_95_ratio)

    max_cvar_95_ratio = population.max(metric=Metrics.CVAR_95_RATIO)
    print(max_cvar_95_ratio.cvar_95_ratio)

    # Composition
    population.plot_composition(names=[max_cdar_95_ratio.name, max_cvar_95_ratio.name])

    # Prices
    population.plot_cumulative_returns(names=[max_cdar_95_ratio.name, max_cvar_95_ratio.name])
