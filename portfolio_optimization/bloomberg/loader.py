import logging
import datetime as dt
import pandas as pd
from xbbg import blp

logger = logging.getLogger('portfolio_optimization.loader')

__all__ = ['save_bloomberg_prices',
           'load_bloomberg_prices']

PRICE_FILE = r'./data/prices.csv'


def save_bloomberg_prices(date_from: dt.date, date_to: dt.date):
    """
    Load bloomberg prices for each bloomberg ticker from tickers.csv and save them in prices.csv
    """
    tickers_df = pd.read_csv(r'./data/tickers.csv', sep=',')
    tickers = list(tickers_df['ticker'].unique())
    df = blp.bdh(tickers, 'px_last', date_from, date_to)
    df.columns = [x[0] for x in df.columns]
    df.to_csv(PRICE_FILE, sep=',')
    logger.info(f'Bloomberg prices saved in {PRICE_FILE}')


def load_bloomberg_prices(date_from: dt.date, date_to: dt.date = None) -> pd.DataFrame:
    """
    Read bloomberg prices saved in prices.csv and return a DataFrame
    """
    df = pd.read_csv(PRICE_FILE, sep=',', index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return df.loc[date_from:date_to]


if __name__ == '__main__':
    # save_bloomberg_prices(date_from=dt.date(2018, 1, 1), date_to=dt.date.today())
    print(load_bloomberg_prices(date_from=dt.date(2018, 1, 1)))
