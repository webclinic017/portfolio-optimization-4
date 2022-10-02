from typing import Union, Optional
import pandas as pd
import plotly.express as px
import numpy as np
from numba import jit

from portfolio_optimization.meta import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.utils.tools import *

__all__ = ['Population']


class Population:
    def __init__(self, portfolios: list[Union[Portfolio, MultiPeriodPortfolio]] = None):
        if portfolios is None:
            portfolios = []
        self.portfolios = portfolios
        self.hashmap = {p.name: p for p in self.portfolios}
        self._fronts = None

    def non_denominated_sort(self, first_front_only: bool = False) -> list[list[int]]:
        """ Fast non-dominated sorting.
        Sort the portfolios into different non-domination levels.
        Complexity O(MN^2) where M is the number of objectives and N the number of portfolios.
        :param first_front_only: If :obj:`True` sort only the first front and exit.
        :returns: A list of Pareto fronts (lists), the first list includes non-dominated portfolios.
        """
        fronts = []
        n = len(self.portfolios)

        if n == 0:
            return fronts

        # final rank that will be returned
        n_ranked = 0
        ranked = np.zeros(n, dtype=int)

        # for each portfolio a list of all portfolios that are dominated by this one
        is_dominating = [[] for _ in range(n)]

        # storage for the number of solutions dominated this one
        n_dominated = np.zeros(n)

        current_front = []

        for i in range(n):
            for j in range(i + 1, n):
                if self.portfolios[i].dominates(self.portfolios[j]):
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif self.portfolios[j].dominates(self.portfolios[i]):
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1

        # append the first front to the current front
        fronts.append(current_front)

        if first_front_only:
            return fronts

        # while not all solutions are assigned to a pareto front
        while n_ranked < n:
            next_front = []
            # for each portfolio in the current front
            for i in current_front:
                # all solutions that are dominated by this portfolio
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0
                        n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts

    def non_denominated_sort_numba(self, first_front_only: bool = False) -> list[list[int]]:
        """ Fast non-dominated sorting.
        Sort the portfolios into different non-domination levels.
        Complexity O(MN^2) where M is the number of objectives and N the number of portfolios.
        :param first_front_only: If :obj:`True` sort only the first front and exit.
        :returns: A list of Pareto fronts (lists), the first list includes non-dominated portfolios.
        """
        n = len(self.portfolios)

        fitnesses = np.array([self.portfolios[i].fitness for i in range(n)])

        fronts = non_denominated_sort(n=n, fitnesses=fitnesses, first_front_only=first_front_only)
        return fronts

    @property
    def fronts(self) -> list[list[int]]:
        if self._fronts is None:
            self._fronts = self.non_denominated_sort()
        return self._fronts

    @property
    def fronts_numba(self) -> list[list[int]]:
        if self._fronts is None:
            self._fronts = self.non_denominated_sort_numba()
        return self._fronts

    @property
    def length(self) -> int:
        return len(self.portfolios)

    def add(self, portfolio: Union[Portfolio, MultiPeriodPortfolio]):
        if portfolio.name in self.hashmap.keys():
            raise KeyError(f'portfolio id {portfolio.name} is already in the population')
        self.portfolios.append(portfolio)
        self.hashmap[portfolio.name] = portfolio

    def get(self, name: str) -> Union[Portfolio, MultiPeriodPortfolio]:
        return self.hashmap[name]

    def iloc(self, i: int) -> Union[Portfolio, MultiPeriodPortfolio]:
        return self.portfolios[i]

    def get_portfolios(self,
                       names: Optional[Union[str, list[str]]] = None,
                       tags: Optional[Union[str, list[str]]] = None) -> list[Union[Portfolio, MultiPeriodPortfolio]]:
        if tags is None and names is None:
            return self.portfolios
        if names is not None:
            if isinstance(names, str):
                names = [names]
            else:
                # check for duplicates
                unique, count = np.unique(names, return_counts=True)
                duplicates = unique[count > 1]
                if len(duplicates) > 0:
                    raise ValueError(f'names contains duplicates {list(duplicates)}')
            return [self.get(name) for name in names]
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            return [portfolio for portfolio in self.portfolios if portfolio.tag in tags]

    def sort(self,
             metric: Metrics,
             reverse: bool = False,
             names: Union[str, list[str]] = None,
             tags: Union[str, list[str]] = None) -> list[Union[Portfolio, MultiPeriodPortfolio]]:
        portfolios = self.get_portfolios(names=names, tags=tags)
        return sorted(portfolios, key=lambda x: x.__getattribute__(metric.value), reverse=reverse)

    def k_min(self, metric: Metrics,
              k: int,
              names: Union[str, list[str]] = None,
              tags: Union[str, list[str]] = None) -> list[Union[Portfolio, MultiPeriodPortfolio]]:
        return self.sort(metric=metric, reverse=False, names=names, tags=tags)[:k]

    def k_max(self,
              metric: Metrics,
              k: int,
              names: Union[str, list[str]] = None,
              tags: Union[str, list[str]] = None) -> list[Union[Portfolio, MultiPeriodPortfolio]]:
        return self.sort(metric=metric, reverse=True, names=names, tags=tags)[:k]

    def min(self,
            metric: Metrics,
            names: Union[str, list[str]] = None,
            tags: Union[str, list[str]] = None) -> Union[Portfolio, MultiPeriodPortfolio]:
        return self.sort(metric=metric, reverse=False, names=names, tags=tags)[0]

    def max(self,
            metric: Metrics,
            names: Union[str, list[str]] = None,
            tags: Union[str, list[str]] = None) -> Union[Portfolio, MultiPeriodPortfolio]:
        return self.sort(metric=metric, reverse=True, names=names, tags=tags)[0]

    def composition(self,
                    names: Union[str, list[str]] = None,
                    tags: Union[str, list[str]] = None) -> pd.DataFrame:
        portfolios = self.get_portfolios(names=names, tags=tags)
        comp_list = []
        for p in portfolios:
            if isinstance(p, MultiPeriodPortfolio):
                comp = p.composition
                comp.rename(columns={c: f'{p.name}_{c}' for c in comp.columns}, inplace=True)
                comp_list.append(comp)
            else:
                comp_list.append(p.composition)

        df = pd.concat(comp_list, axis=1)
        df.fillna(0, inplace=True)
        return df

    def plot_cumulative_returns(self, idx=slice(None),
                                names: Union[str, list[str]] = None,
                                tags: Union[str, list[str]] = None,
                                show: bool = True):
        portfolios = self.get_portfolios(names=names, tags=tags)
        df = pd.concat([p.cumulative_returns_df for p in portfolios], axis=1).iloc[:, idx]
        df.columns = [p.name for p in portfolios]
        fig = df.plot()
        fig.update_layout(title='Prices',
                          xaxis_title='Dates',
                          yaxis_title='Prices',
                          legend_title_text='Portfolios')
        if show:
            fig.show()
        else:
            return fig

    def plot_composition(self,
                         names: Union[str, list[str]] = None,
                         tags: Union[str, list[str]] = None,
                         show: bool = True):
        df = self.composition(names=names, tags=tags).T
        fig = px.bar(df, x=df.index, y=df.columns, title='Portfolios Composition')
        if show:
            fig.show()
        else:
            return fig

    def plot_metrics(self,
                     x: Metrics,
                     y: Metrics,
                     z: Metrics = None,
                     hover_metrics: list[Metrics] = None,
                     fronts: bool = False,
                     color_scale: Union[Metrics, str] = None,
                     names: Union[str, list[str]] = None,
                     tags: Union[str, list[str]] = None,
                     title='Portfolios',
                     show: bool = True):
        portfolios = self.get_portfolios(names=names, tags=tags)
        num_fmt = ':.3f'
        hover_data = {x.value: num_fmt,
                      y.value: num_fmt,
                      'tag': True}

        if z is not None:
            hover_data[z.value] = num_fmt

        if hover_metrics is not None:
            for metric in hover_metrics:
                hover_data[metric.value] = num_fmt

        columns = list(hover_data.keys())
        columns.append('name')
        if isinstance(color_scale, Metrics):
            color_scale = color_scale.value
            hover_data[color_scale] = num_fmt

        if color_scale is not None and color_scale not in columns:
            columns.append(color_scale)

        res = [[portfolio.__getattribute__(attr) for attr in columns] for portfolio in portfolios]
        df = pd.DataFrame(res, columns=columns)
        if fronts:
            if tags is not None:
                ValueError(f'Cannot plot front with tags selected')
            df['front'] = -1
            for i, front in enumerate(self.fronts):
                for idx in front:
                    df.iloc[idx, -1] = str(i)
            color = df.columns[-1]
        elif color_scale is not None:
            color = color_scale
        else:
            color = 'tag'

        if z is not None:
            fig = px.scatter_3d(df,
                                x=x.value,
                                y=y.value,
                                z=z.value,
                                hover_name='name',
                                hover_data=hover_data,
                                color=color,
                                symbol='tag')
        else:
            fig = px.scatter(df,
                             x=x.value,
                             y=y.value,
                             hover_name='name',
                             hover_data=hover_data,
                             color=color,
                             symbol='tag')
        fig.update_traces(marker_size=10)
        fig.update_layout(title=title,
                          legend=dict(yanchor='top',
                                      y=0.99,
                                      xanchor='left',
                                      x=1.15))
        if show:
            fig.show()
        else:
            return fig

    def __str__(self):
        return f'Population <{len(self.portfolios)} portfolios>'

    def __repr__(self):
        return str(self)
