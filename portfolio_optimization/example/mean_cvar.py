import datetime as dt

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization import *
from portfolio_optimization.loader import *
from portfolio_optimization.bloomberg.loader import *


def mean_variance_vs_mean_cvar():
    """
    Compare the Efficient Frontier of the mean-variance against the mean-cvar optimization
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

    # Efficient Frontier -- Mean Variance

    portfolios_weights = model.mean_variance(population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 assets=assets,
                                 name=f'mean_variance_{i}',
                                 tag='mean_variance'))

    # Efficient Frontier -- Mean CVaR
    portfolios_weights = model.mean_cvar(beta=0.95,
                                         population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 assets=assets,
                                 name=f'mean_cvar_{i}',
                                 tag='mean_cvar'))

    # Plot
    population.plot_metrics(x=Metrics.ANNUALIZED_STD,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.SHARPE_RATIO)
    population.plot_metrics(x=Metrics.CDAR_95,
                            y=Metrics.ANNUALIZED_MEAN,
                            color_scale=Metrics.CVAR_95_RATIO)

    # Metrics
    max_sharpe = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe.sharpe_ratio)

    max_cvar_95_ratio = population.max(metric=Metrics.CVAR_95_RATIO)
    print(max_cvar_95_ratio.cvar_95_ratio)

    # Composition
    population.plot_composition(names=[max_sharpe.name, max_cvar_95_ratio.name])

    # Prices
    population.plot_cumulative_returns(names=[max_sharpe.name, max_cvar_95_ratio.name])
