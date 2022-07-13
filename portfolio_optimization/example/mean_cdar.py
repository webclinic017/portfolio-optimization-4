import datetime as dt

from portfolio_optimization.meta import FitnessType, InvestmentType, Metrics
from portfolio_optimization.paths import EXAMPLE_PRICES_PATH
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.population import Population
from portfolio_optimization.optimization import mean_variance, mean_cdar, mean_cvar
from portfolio_optimization.loader import load_assets
from portfolio_optimization.bloomberg.loader import load_prices


def mean_variance_vs_mean_cdar():
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
                                 name=f'mean_variance_{i}',
                                 tag='mean_variance'))

    # Efficient Frontier -- Mean CDaR
    portfolios_weights = mean_cdar(expected_returns=assets.mu,
                                   returns=assets.returns,
                                   investment_type=InvestmentType.FULLY_INVESTED,
                                   weight_bounds=(0, None),
                                   beta=0.95,
                                   population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 name=f'mean_cdar_{i}',
                                 tag='mean_cdar'))

    # Plot
    population.plot(x=Metrics.ANNUALIZED_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SHARPE_RATIO)
    population.plot(x=Metrics.ANNUALIZED_DOWNSIDE_STD,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.SORTINO_RATIO)
    population.plot(x=Metrics.MAX_DRAWDOWN,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.CALMAR_RATIO)
    population.plot(x=Metrics.CDAR_95,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.CDAR_95_RATIO)

    # Metrics
    max_sharpe = population.max(metric=Metrics.SHARPE_RATIO)
    print(max_sharpe.sharpe_ratio)

    max_cdar_95_ratio = population.max(metric=Metrics.CDAR_95_RATIO)
    print(max_cdar_95_ratio.cdar_95_ratio)

    # Composition
    population.plot_composition(names=[max_sharpe.name, max_cdar_95_ratio.name])


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

    # Efficient Frontier -- Mean CDaR
    portfolios_weights = mean_cdar(expected_returns=assets.mu,
                                   returns=assets.returns,
                                   investment_type=InvestmentType.FULLY_INVESTED,
                                   weight_bounds=(0, None),
                                   beta=0.95,
                                   population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 name=f'mean_cdar_{i}',
                                 tag='mean_cdar'))

    # Efficient Frontier -- Mean CVaR
    portfolios_weights = mean_cvar(expected_returns=assets.mu,
                                   returns=assets.returns,
                                   investment_type=InvestmentType.FULLY_INVESTED,
                                   weight_bounds=(0, None),
                                   beta=0.95,
                                   population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.add(Portfolio(weights=weights,
                                 fitness_type=FitnessType.MEAN_STD,
                                 assets=assets,
                                 name=f'mean_cvar_{i}',
                                 tag='mean_cvar'))

    # Plot
    population.plot(x=Metrics.CDAR_95,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.CDAR_95_RATIO)
    population.plot(x=Metrics.CVAR_95,
                    y=Metrics.ANNUALIZED_MEAN,
                    color_scale=Metrics.CVAR_95_RATIO)

    # Metrics
    max_cdar_95_ratio = population.max(metric=Metrics.CDAR_95_RATIO)
    print(max_cdar_95_ratio.cdar_95_ratio)

    max_cvar_95_ratio = population.max(metric=Metrics.CVAR_95_RATIO)
    print(max_cvar_95_ratio.cvar_95_ratio)

    # Composition
    population.plot_composition(names=[max_cdar_95_ratio.name, max_cvar_95_ratio.name])
