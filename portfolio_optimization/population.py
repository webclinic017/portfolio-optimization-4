import logging
from typing import Union, Optional, Dict
from itertools import islice
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from functools import cached_property
from collections.abc import Iterator

from portfolio_optimization.meta import *
from portfolio_optimization.portfolio import *
from portfolio_optimization.utils.sorting import *

__all__ = ['Population']

logger = logging.getLogger('portfolio_optimization.population')


class Population:
    def __init__(self, portfolios: Optional[list[BasePortfolio]] = None):
        self.hashmap = self._hashmap(portfolios=portfolios)

    @staticmethod
    def _hashmap(portfolios: Optional[list[BasePortfolio]]) -> Dict[str, BasePortfolio]:
        hashmap = {}
        if portfolios is not None:
            for p in portfolios:
                if not isinstance(p, BasePortfolio):
                    raise TypeError(f'Portfolio has wrong type {type(p)}')
                if p.name in hashmap:
                    raise KeyError(f'portfolio {p.name} is in duplicate')
                hashmap[p.name] = p
        return hashmap

    @property
    def portfolios(self) -> list[BasePortfolio]:
        return list(self.hashmap.values())

    @portfolios.setter
    def portfolios(self, value: Optional[list[BasePortfolio]] = None):
        self.hashmap = self._hashmap(portfolios=value)

    def __len__(self) -> int:
        return len(self.hashmap)

    def __getitem__(self, key: Union[int, slice]) -> Union[BasePortfolio, list[BasePortfolio]]:
        if isinstance(key, slice) or key < 0:
            return list(self.hashmap.values())[key]
        return self.hashmap[next(islice(self.hashmap, key, None))]

    def __setitem__(self, key: int, value: BasePortfolio) -> None:
        if not isinstance(value, BasePortfolio):
            raise TypeError(f'Cannot set a value with type {type(value)}')
        new_name = value.name
        old_name = list(self.hashmap.keys())[key]
        if new_name == old_name:
            self.hashmap[old_name] = value
        else:
            if new_name in self.hashmap:
                raise KeyError(f'portfolio {new_name} is already in the population')
            # Create a new dict to preserve the order
            self.hashmap[old_name] = value
            self.hashmap = {k if k != old_name else new_name: v for k, v in self.hashmap.items()}

    def __delitem__(self, key: int) -> None:
        del self.hashmap[next(islice(self.hashmap, key, None))]

    def __iter__(self) -> Iterator[BasePortfolio]:
        return iter(self.hashmap.values())

    def __contains__(self, value: BasePortfolio) -> bool:
        if not isinstance(value, BasePortfolio):
            raise False
        return value.name in self.hashmap

    def append(self, value: BasePortfolio) -> None:
        if not isinstance(value, BasePortfolio):
            raise TypeError(f'Cannot append a value with type {type(value)}')
        if value.name in self.hashmap:
            raise KeyError(f'portfolio {value.name} is already in the population')
        self.hashmap[value.name] = value

    def get(self, name: str) -> BasePortfolio:
        try:
            return self.hashmap[name]
        except KeyError:
            raise KeyError(f'No portfolio found with name {name}')

    def non_denominated_sort(self, first_front_only: bool = False) -> list[list[int]]:
        """ Fast non-dominated sorting.
        Sort the portfolios into different non-domination levels.
        Complexity O(MN^2) where M is the number of objectives and N the number of portfolios.
        :param first_front_only: If :obj:`True` sort only the first front and exit.
        :returns: A list of Pareto fronts (lists), the first list includes non-dominated portfolios.
        """
        n = len(self)
        fitnesses = np.array([portfolio.fitness for portfolio in self])
        fronts = non_denominated_sort(n=n, fitnesses=fitnesses, first_front_only=first_front_only)
        return fronts

    @cached_property
    def fronts(self) -> list[list[int]]:
        return self.non_denominated_sort()

    def get_portfolios(self,
                       names: Optional[Union[str, list[str]]] = None,
                       tags: Optional[Union[str, list[str]]] = None) -> list[BasePortfolio]:
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
            return [portfolio for portfolio in self if portfolio.tag in tags]

    def sort(self,
             metric: Metrics,
             reverse: bool = False,
             names: Union[str, list[str]] = None,
             tags: Union[str, list[str]] = None) -> list[BasePortfolio]:
        portfolios = self.get_portfolios(names=names, tags=tags)
        return sorted(portfolios, key=lambda x: x.__getattribute__(metric.value), reverse=reverse)

    def k_min(self, metric: Metrics,
              k: int,
              names: Union[str, list[str]] = None,
              tags: Union[str, list[str]] = None) -> list[BasePortfolio]:
        return self.sort(metric=metric, reverse=False, names=names, tags=tags)[:k]

    def k_max(self,
              metric: Metrics,
              k: int,
              names: Union[str, list[str]] = None,
              tags: Union[str, list[str]] = None) -> list[BasePortfolio]:
        return self.sort(metric=metric, reverse=True, names=names, tags=tags)[:k]

    def min(self,
            metric: Metrics,
            names: Union[str, list[str]] = None,
            tags: Union[str, list[str]] = None) -> BasePortfolio:
        return self.sort(metric=metric, reverse=False, names=names, tags=tags)[0]

    def max(self,
            metric: Metrics,
            names: Union[str, list[str]] = None,
            tags: Union[str, list[str]] = None) -> BasePortfolio:
        return self.sort(metric=metric, reverse=True, names=names, tags=tags)[0]

    def summary(self,
                names: Union[str, list[str]] = None,
                tags: Union[str, list[str]] = None,
                formatted: bool = True) -> pd.DataFrame:
        portfolios = self.get_portfolios(names=names, tags=tags)
        return pd.concat([p.summary(formatted=formatted) for p in portfolios], keys=[p.name for p in portfolios],
                         axis=1)

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
                                show: bool = True) -> Optional[go.Figure]:
        portfolios = self.get_portfolios(names=names, tags=tags)
        df = pd.concat([p.cumulative_returns_df for p in portfolios], axis=1).iloc[:, idx]
        df.columns = [p.name for p in portfolios]
        fig = df.plot()
        fig.update_layout(title='Cumulative Returns',
                          xaxis_title='Dates',
                          yaxis_title='Cumulative Returns (%)',
                          legend_title_text='Portfolios')
        if show:
            fig.show()
        else:
            return fig

    def plot_composition(self,
                         names: Union[str, list[str]] = None,
                         tags: Union[str, list[str]] = None,
                         show: bool = True) -> Optional[go.Figure]:
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
                     show: bool = True) -> Optional[go.Figure]:
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
