from pathlib import Path

__all__ = ['TEST_FOLDER',
           'TEST_PRICES_PATH',
           'EXAMPLE_FOLDER',
           'EXAMPLE_PRICES_PATH']

ROOT = Path(Path().resolve(), 'portfolio_optimization')
TEST_FOLDER = Path(ROOT, 'test')
TEST_PRICES_PATH = Path(TEST_FOLDER, 'data', 'stock_prices.csv')
EXAMPLE_FOLDER = Path(ROOT, 'example')
EXAMPLE_PRICES_PATH = Path(EXAMPLE_FOLDER, 'data', 'prices.csv')
EXAMPLE_TICKERS_PATH = Path(EXAMPLE_FOLDER, 'data', 'tickers.csv')
