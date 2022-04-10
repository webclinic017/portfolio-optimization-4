import logging
import datetime as dt

today = dt.datetime.today()
logger = logging.getLogger('portfolio_optimization')
if len(logger.handlers) == 0:
    logger.setLevel(logging.INFO)
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.INFO)
    logger.addHandler(console_log)
