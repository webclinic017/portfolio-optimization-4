import datetime as dt
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.express as px

from portfolio_optimization.bloomberg.loader import *
from portfolio_optimization.utils.preprocessing import *
# (*) To communicate with Plotly's server, sign in with credentials file
import matplotlib.pyplot as plt

pd.options.plotting.backend = "plotly"

np.random.seed(150)


def rand_weights(n: int) -> np.array:
    """
    Produces n random weights that sum to 1
    """
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(mu: np.matrix, cov: np.matrix) -> np.array:
    """
    Returns the mean and standard deviation of returns for a random portfolio
    """
    w = np.asmatrix(rand_weights(n=len(mu)))
    ptf_mu = w @ mu
    ptf_sigma = np.sqrt(w @ cov @ w.T)
    return np.array([ptf_mu[0, 0], ptf_sigma[0, 0]])


def markowitz_optimization(mu: np.matrix, cov: np.matrix, sample: int) -> (np.array, np.array):
    """
    Markowitz optimization:
    Constraints: No-short selling and portfolio invested at 100%
    :param mu:
    :param cov:
    :param sample:
    :return:
    """
    w = cp.Variable(len(mu))
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, cov)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk),
                      [cp.sum(w) == 1,
                       w >= 0])

    portfolios = []
    weights = []
    for value in np.logspace(-2, 3, num=sample):
        gamma.value = value
        prob.solve()
        portfolios.append([ret.value[0], np.sqrt(risk.value)])
        weights.append(w.value)

    return np.array(portfolios), np.array(weights)


def portfolio_with_optimal_sharp(mu: np.matrix, cov: np.matrix, portfolios: np.array):
    # Compute the second degree polynomial of the pareto front
    m1 = np.polyfit(portfolios[:, 0], portfolios[:, 1], 2)
    target_ret = np.sqrt(m1[2] / m1[0])
    # compute the portfolio with optimal sharp ratio
    w = cp.Variable(len(mu))
    ret = mu.T @ w
    risk = cp.quad_form(w, cov)
    prob = cp.Problem(cp.Minimize(risk),
                      [ret == target_ret,
                       cp.sum(w) == 1,
                       w >= 0])
    prob.solve()

    return np.array([ret.value[0], np.sqrt(risk.value)])


def annualize(portfolios: np.array, percent: bool = True) -> np.array:
    n = 255
    ptf = np.copy(portfolios)
    ptf[:, 0] = ptf[:, 0] * n
    ptf[:, 1] = ptf[:, 1] * np.sqrt(n)
    if percent:
        ptf = ptf * 100
    return ptf


def plot(random_portfolios: np.array, optimal_portfolios: np.array):
    random_portfolios = annualize(portfolios=random_portfolios)
    optimal_portfolios = annualize(portfolios=optimal_portfolios)

    fig = px.scatter(x=random_portfolios[:, 1],
                     y=random_portfolios[:, 0])
    fig.add_scatter(x=optimal_portfolios[:, 1], y=optimal_portfolios[:, 0],
                    mode='markers',
                    marker=dict(
                        size=10, color=optimal_portfolios[:, 0] / optimal_portfolios[:, 1],
                        colorbar=dict(
                            title='Sharp Ratio'
                        ),
                        colorscale='Viridis'
                    ),
                    name='Pareto Optimal Portfolios')
    fig.update_layout(
        title='Portfolios',
        xaxis_title='Standard deviation',
        yaxis_title="Return")
    fig.update_yaxes(ticksuffix='%')
    fig.update_xaxes(ticksuffix='%')
    fig.show()


def run():
    prices = load_bloomberg_prices(date_from=dt.date(2019, 1, 1))
    daily_returns = preprocessing(prices=prices)
    cum_returns = (daily_returns + 1).cumprod()

    returns = daily_returns[:100, :]
    mu = np.asmatrix(np.mean(returns, axis=1)).T
    cov = np.asmatrix(np.cov(returns))

    random_portfolios = np.array([random_portfolio(mu=mu, cov=cov) for _ in range(10000)])
    optimal_portfolios, weights = markowitz_optimization(mu=mu, cov=cov, sample=100)
    plot(random_portfolios=random_portfolios, optimal_portfolios=optimal_portfolios)

    ptf_optimal_sharp = portfolio_with_optimal_sharp(mu=mu, cov=cov, portfolios=optimal_portfolios)
    annualize(ptf_optimal_sharp)

    fig = cum_returns.iloc[:, 210:215].plot(title='Prices')
    fig.show()
