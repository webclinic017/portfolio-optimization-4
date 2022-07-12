import datetime as dt

from portfolio_optimization.utils.tools import *


def test_walk_forward():
    start_date = dt.date(2018, 1, 1)
    end_date = dt.date(2019, 1, 10)
    train_duration = 150
    test_duration = 30
    prev_test_end = start_date + dt.timedelta(days=train_duration)
    for (train_start, train_end), (test_start, test_end) in walk_forward(start_date=start_date,
                                                                         end_date=end_date,
                                                                         train_duration=train_duration,
                                                                         test_duration=test_duration,
                                                                         full_period=False):
        assert (train_end - train_start).days == train_duration
        assert 1 <= (test_end - test_start).days <= test_duration
        assert test_start == train_end
        assert test_start == prev_test_end
        prev_test_end = test_end
