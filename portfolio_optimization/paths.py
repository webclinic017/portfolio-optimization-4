from pathlib import Path

__all__ = ['TEST_FOLDER',
           'TEST_PRICES_PATH']

ROOT = Path().resolve()
TEST_FOLDER = Path(ROOT, 'portfolio_optimization', 'test')
TEST_PRICES_PATH = Path(TEST_FOLDER, 'data', 'prices.csv')
