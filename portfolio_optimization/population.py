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
        self._fronts = None

    def sort(self, first_front_only: bool = False):
        return fast_nondominated_sorting(portfolios=self.portfolios, first_front_only=first_front_only)

    @property
    def fronts(self):
        if self._fronts is None:
            self._fronts = self.sort()
        return self._fronts

    @property
    def length(self):
        return len(self.portfolios)

    def append(self, portfolio: Portfolio):
        self.portfolios.append(portfolio)

    def plot(self, x: str, y: str, z: str = None, fronts: bool = True):
        columns = [x, y]
        if z is not None:
            columns.append(z)

        color = None
        res = [[portfolio.__getattribute__(attr) for attr in columns] for portfolio in self.portfolios]
        df = pd.DataFrame(res, columns=columns)
        if fronts:
            df['front'] = -1
            for i, front in enumerate(self.fronts):
                for idx in front:
                    df.iloc[idx, -1] = str(i)
            color = df.columns[-1]

        if len(columns) == 3:
            fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, symbol=color)
        else:
            fig = px.scatter(df, x=x, y=y, color=color, symbol=color)
        fig.update_traces(marker_size=10)
        fig.show()

    def __str__(self):
        return f'Population ({len(self.portfolios)} portfolios)'

    def __repr__(self):
        return str(self)


