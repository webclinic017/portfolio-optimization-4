import datetime as dt

from portfolio_optimization.meta import *
from portfolio_optimization.paths import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.population import *
from portfolio_optimization.optimization import *
from portfolio_optimization.loader import *
from portfolio_optimization.utils.tools import *

if __name__ == '__main__':
    """
    Compare the Efficient Frontier of the mean-variance against the mean-cvar optimization
    """
    prices = load_prices(file=EXAMPLE_PRICES_PATH)

    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         pre_selection_number=50,
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

    # Efficient Frontier -- Mean CVaR
    portfolios_weights = model.mean_cvar(beta=0.95,
                                         population_size=30)
    for i, weights in enumerate(portfolios_weights):
        population.append(Portfolio(weights=weights,
                                    assets=assets,
                                    name=f'mean_cvar_{i}',
                                    tag='mean_cvar'))

    # Plot
    population.plot_metrics(x=Metric.ANNUALIZED_STD,
                            y=Metric.ANNUALIZED_MEAN,
                            color_scale=Metric.SHARPE_RATIO)
    population.plot_metrics(x=Metric.CDAR,
                            y=Metric.ANNUALIZED_MEAN,
                            color_scale=Metric.CVAR_RATIO)

    # Metrics
    max_sharpe_ptf = population.max(metric=Metric.SHARPE_RATIO)
    print(max_sharpe_ptf.sharpe_ratio)
    print(max_sharpe_ptf.summary())

    max_cvar_ratio_ptf = population.max(metric=Metric.CVAR_RATIO)
    print(max_cvar_ratio_ptf.cvar_ratio)

    portfolio_names = [max_sharpe_ptf.name, max_cvar_ratio_ptf.name]
    # Composition
    population.plot_composition(names=portfolio_names)
    print(max_cvar_ratio_ptf.assets_names)

    # Cumulative returns
    population.plot_cumulative_returns(names=portfolio_names)

    # Summary
    print(population.summary(names=portfolio_names))
    print(population.summary())

    # Plot compo assets
    assets.plot_cumulative_returns(names=max_cvar_ratio_ptf.assets_names)

