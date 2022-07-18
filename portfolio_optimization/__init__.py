import logging
from .assets import Assets
from .portfolio import Portfolio, MultiPeriodPortfolio
from .population import Population
from .loader import load_assets, load_train_test_assets
from .utils import walk_forward
from .optimization import *
from .bloomberg import *
from .exception import *

from .paths import *
from .meta import *

__all__ = ['Assets',
           'Portfolio',
           'MultiPeriodPortfolio',
           'Population',
           'load_assets',
           'load_train_test_assets',
           'walk_forward']

__all__ += paths.__all__
__all__ += meta.__all__
__all__ += optimization.__all__
__all__ += bloomberg.__all__
__all__ += exception.__all__


class CustomFormatter(logging.Formatter):
    """Logging colored formatter"""

    blue = '\x1b[38;5;39m'
    green = '\x1b[32;1m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.blue + self.fmt + self.reset,
            logging.INFO: self.green + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger('portfolio_optimization')
if len(logger.handlers) == 0:
    logger.setLevel(logging.INFO)
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.INFO)
    formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_log.setFormatter(formatter)
    logger.addHandler(console_log)
