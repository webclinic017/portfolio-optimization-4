import pandas as pd
import plotly.express as px

from portfolio_optimization.portfolio import *
from portfolio_optimization.utils.sorting import *

__all__ = ['Population']


class Population:
    def __init__(self, portfolios: list[Portfolio] = None):
        if portfolios is None:
            portfolios = []
        self.portfolios = portfolios
        self.hashmap = {p.pid: p for p in self.portfolios}
        self._fronts = None

    def non_denominated_sort(self, first_front_only: bool = False) -> list[list[int]]:
        return fast_nondominated_sorting(portfolios=self.portfolios, first_front_only=first_front_only)

    @property
    def fronts(self) -> list[list[int]]:
        if self._fronts is None:
            self._fronts = self.non_denominated_sort()
        return self._fronts

    @property
    def length(self) -> int:
        return len(self.portfolios)

    def add(self, portfolio: Portfolio):
        if portfolio.pid in self.hashmap.keys():
            raise KeyError(f'portfolio id {portfolio.pid} is already in the population')
        self.portfolios.append(portfolio)
        self.hashmap[portfolio.pid] = portfolio

    def get(self, pid: str) -> Portfolio:
        return self.hashmap[pid]

    def iloc(self, i: int) -> Portfolio:
        return self.portfolios[i]

    def get_portfolios_by_tag(self, tag: str) -> list[Portfolio]:
        return [portfolio for portfolio in self.portfolios if portfolio.tag == tag]

    def sort(self, metric: Metrics, reverse: bool = False, tag: str = None) -> list[Portfolio]:
        if tag is None:
            portfolios = self.portfolios
        else:
            portfolios = self.get_portfolios_by_tag(tag=tag)

        return sorted(portfolios, key=lambda x: x.__getattribute__(metric.value), reverse=reverse)

    def k_min(self, metric: Metrics, k: int, tag: str = None) -> list[Portfolio]:
        return self.sort(metric=metric, reverse=False, tag=tag)[:k]

    def k_max(self, metric: Metrics, k: int, tag: str = None) -> list[Portfolio]:
        return self.sort(metric=metric, reverse=True, tag=tag)[:k]

    def min(self, metric: Metrics, tag: str = None) -> Portfolio:
        return self.sort(metric=metric, reverse=False, tag=tag)[0]

    def max(self, metric: Metrics, tag: str = None) -> Portfolio:
        return self.sort(metric=metric, reverse=True, tag=tag)[0]

    def composition(self) -> pd.DataFrame:
        res = []
        idx = []
        for pid, p in self.hashmap.items():
            res.append(p.composition.to_dict()['weight'])
            idx.append(pid)
        df = pd.DataFrame(res, index=idx)
        df.index.name = 'portfolio'
        return df

    def plot_composition(self):
        df = self.composition()
        fig = px.bar(df, x=df.index, y=df.columns, title='Portfolios Composition')
        fig.show()

    def plot(self, x: Metrics, y: Metrics, z: Metrics = None, fronts: bool = False, color_scale: str = None):
        columns = [x.value, y.value, 'tag']
        if z is not None:
            columns.append(z.value)
        if color_scale is not None:
            columns.append(color_scale)

        res = [[portfolio.__getattribute__(attr) for attr in columns] for portfolio in self.portfolios]
        df = pd.DataFrame(res, columns=columns)
        if fronts:
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
            fig = px.scatter_3d(df, x=x.value, y=y.value, z=z.value, color=color, symbol='tag')
        else:
            fig = px.scatter(df, x=x.value, y=y.value, color=color, symbol='tag')
        fig.update_traces(marker_size=10)
        fig.update_layout(legend=dict(
            yanchor='top',
            y=0.995,
            xanchor='left',
            x=1.1
        ))
        fig.show()

    def __str__(self):
        return f'Population ({len(self.portfolios)} portfolios)'

    def __repr__(self):
        return str(self)
