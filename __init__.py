import logging
import datetime as dt

today = dt.datetime.today()
logger = logging.getLogger('portfolio-optimization')
print('coucou')
if len(logger.handlers) == 0:
    logger.setLevel(logging.INFO)
    # Console logger
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.INFO)
    # console_format = '  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s'
    logger.addHandler(console_log)
