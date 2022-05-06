import datetime as dt
import numpy as np

from portfolio_optimization.assets import *
from portfolio_optimization.utils.assets import *


def test_assets_class():
    assets = Assets(start_date=dt.date(2019, 1, 1))
    names = list(assets.prices.columns)
    assert assets.asset_nb == len(names)
    assert assets.date_nb == len(assets.prices) - 1
    assert assets.names == names
    ret = []
    for i in range(1, len(assets.prices)):
        ret.append((assets.prices.iloc[i] / assets.prices.iloc[i - 1] - 1).to_numpy())
    ret = np.array(ret).T
    assert np.array_equal(ret, assets.returns)
    assert (abs(ret.mean(axis=1) - assets.mu)).sum() < 1e-10
    assert (abs(np.cov(ret) - assets.cov)).sum() < 1e-10

    new_names = [names[i] for i in np.random.choice(len(names), 30, replace=False)]
    assets = Assets(start_date=dt.date(2019, 1, 1), names_to_keep=new_names)
    assert assets.asset_nb == len(new_names)
    assert assets.date_nb == len(assets.prices) - 1
    assets.plot()
    assets.plot(idx=[2, 5])


def test_load_assets():
    correlation_threshold = 0.99
    assets = load_assets(start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=200,
                         correlation_threshold=correlation_threshold,
                         pre_selection_number=100)
    for i in range(assets.asset_nb):
        for j in range(assets.asset_nb):
            if i != j:
                assert assets.corr[i, j] < correlation_threshold
    assert 100 <= len(assets.names) <= 200


def test_load_train_test_assets():
    train_period = (dt.date(2018, 1, 1), dt.date(2019, 1, 1))
    test_period = (dt.date(2019, 1, 1), dt.date(2020, 1, 1))

    train_assets, test_assets = load_train_test_assets(train_period=train_period,
                                                       test_period=test_period,
                                                       random_selection=200,
                                                       pre_selection_number=100)

    assert set(train_assets.names) == set(test_assets.names)
    assert (train_assets.start_date, train_assets.end_date) == train_period
    assert (test_assets.start_date, test_assets.end_date) == test_period
