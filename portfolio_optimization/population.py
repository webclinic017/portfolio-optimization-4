import pandas as pd
import plotly.express as px
from enum import Enum

from portfolio_optimization.portfolio import *
from portfolio_optimization.utils.sorting import *

__all__ = ['Population']


class Population:
    def __init__(self, portfolios: list[Portfolio] = None):
        if portfolios is None:
            portfolios = []
        self.portfolios = portfolios
        self._fronts = None

    def non_denominated_sort(self, first_front_only: bool = False):
        return fast_nondominated_sorting(portfolios=self.portfolios, first_front_only=first_front_only)

    @property
    def fronts(self):
        if self._fronts is None:
            self._fronts = self.non_denominated_sort()
        return self._fronts

    @property
    def length(self):
        return len(self.portfolios)

    def append(self, portfolio: Portfolio):
        self.portfolios.append(portfolio)

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

    def get_portfolios(self, tag: str):
        return [portfolio for portfolio in self.portfolios if portfolio.tag == tag]

    def sort(self, metric: Metrics, reverse: bool = False, tag: str = None):
        if tag is None:
            portfolios = self.portfolios
        else:
            portfolios = self.get_portfolios(tag=tag)

        return sorted(portfolios, key=lambda x: x.__getattribute__(metric.value), reverse=reverse)

    def k_min(self, metric: Metrics, k: int, tag: str = None):
        return self.sort(metric=metric, reverse=False, tag=tag)[:k]

    def k_max(self, metric: Metrics, k: int, tag: str = None):
        return self.sort(metric=metric, reverse=True, tag=tag)[:k]

    def min(self, metric: Metrics, tag: str = None):
        return self.sort(metric=metric, reverse=False, tag=tag)[0]

    def max(self, metric: Metrics, tag: str = None):
        return self.sort(metric=metric, reverse=True, tag=tag)[0]

    def __str__(self):
        return f'Population ({len(self.portfolios)} portfolios)'

    def __repr__(self):
        return str(self)
