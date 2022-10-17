import logging
from pathlib import Path
import datetime as dt
import pandas as pd
from xbbg import blp

logger = logging.getLogger('portfolio_optimization.loader')

__all__ = ['save_bloomberg_prices']


def save_bloomberg_prices(file: Path | str,
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


if __name__ == '__main__':
    answer = input('Are you sure to reload bloomberg data? [yes/no]')
    if answer == 'yes':
        tickers_df = pd.read_csv(r'./data/tickers.csv', sep=',')
        save_bloomberg_prices(file=r'./data/prices.csv',
                              tickers=list(tickers_df['ticker'].unique()),
                              date_from=dt.date(2018, 1, 1),
                              date_to=dt.date.today())
