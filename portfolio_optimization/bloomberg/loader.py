import logging
from typing import Optional, Union
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
from xbbg import blp

logger = logging.getLogger('portfolio_optimization.loader')

__all__ = ['save_bloomberg_prices',
           'load_prices']


def save_bloomberg_prices(file: Union[Path, str],
                          tickers: list[str],
                          date_from: dt.date,
                          date_to: dt.date):
    """
    Load bloomberg prices for each bloomberg ticker from tickers.csv and save them in prices.csv
    :param file: the path of the prices csv file where the bloomberg data will be saved
    :param tickers: list of bloomberg tickers
    :param date_from: starting date
    :param date_to: ending date
    """
    df = blp.bdh(tickers, 'px_last', date_from, date_to)
    df.columns = [x[0] for x in df.columns]
    df.to_csv(file, sep=',')
    logger.info(f'Bloomberg prices saved in {file}')


def load_prices(file: Union[Path, str],
                date_from: Optional[dt.date] = None,
                date_to: Optional[dt.date] = None,
                names_to_keep: Optional[list[str]] = None,
                random_selection: Optional[int] = None) -> pd.DataFrame:
    """
    Read bloomberg prices saved in prices.csv and return a DataFrame
    :param file: the path of the prices csv file
    :param date_from: starting date
    :param date_to: ending date
    :param names_to_keep: asset names to keep in the final DataFrame
    :param random_selection: number of assets to randomly keep in the final DataFrame
    """
    df = pd.read_csv(file, sep=',', index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    df = df.loc[date_from:date_to]

    if names_to_keep is not None:
        df = df[names_to_keep]

    if random_selection is not None:
        names = list(df.columns)
        new_names = [names[i] for i in np.random.choice(len(names), random_selection, replace=False)]
        df = df[new_names]

    return df


if __name__ == '__main__':
    answer = input('Are you sure to reload bloomberg data? [yes/no]')
    if answer == 'yes':
        tickers_df = pd.read_csv(r'./data/tickers.csv', sep=',')
        save_bloomberg_prices(file=r'./data/prices.csv',
                              tickers=list(tickers_df['ticker'].unique()),
                              date_from=dt.date(2018, 1, 1),
                              date_to=dt.date.today())
        print(load_prices(file=r'./data/prices.csv'))
