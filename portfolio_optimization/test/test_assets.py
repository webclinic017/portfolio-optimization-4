import datetime as dt
import numpy as np

from portfolio_optimization import (Assets,
                                    load_prices,
                                    load_assets,
                                    load_train_test_assets,
                                    TEST_PRICES_PATH)


def test_assets_class():
    prices = load_prices(file=TEST_PRICES_PATH).iloc[:, :30]

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    asset_missing_threshold=0.1,
                    dates_missing_threshold=0.1,
                    correlation_threshold=0.99,
                    verbose=False)

    names = np.array(assets.prices.columns)
    assert assets.asset_nb == len(names)
    assert assets.date_nb == len(assets.prices) - 1
    assert np.array_equal(assets.names, names)
    ret = []
    for i in range(1, len(assets.prices)):
        ret.append((assets.prices.iloc[i] / assets.prices.iloc[i - 1] - 1).to_numpy())
    ret = np.array(ret).T
    assert np.array_equal(ret, assets.returns)
    assert (abs(ret.mean(axis=1) - assets.mu)).sum() < 1e-10
    assert (abs(np.cov(ret) - assets.cov)).sum() < 1e-10

    new_names = [names[i] for i in np.random.choice(len(names), 15, replace=False)]

    assets = Assets(prices=prices,
                    start_date=start_date,
                    names_to_keep=new_names,
                    verbose=False)

    assert assets.asset_nb == len(new_names)
    assert assets.date_nb == len(assets.prices) - 1
    assert assets.plot_cumulative_returns(show=False)
    assert assets.plot_cumulative_returns(idx=[2, 5], show=False)

    costs = {assets.names[2]: 0.1,
             assets.names[10]: 0.2}
    costs_array = assets.dict_to_array(assets_dict=costs)
    for i, name in enumerate(assets.names):
        assert costs.get(name, 0) == costs_array[i]


def test_load_assets():
    prices = load_prices(file=TEST_PRICES_PATH)

    correlation_threshold = 0.99
    random_selection = 15
    pre_selection_number = 10

    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=random_selection,
                         removal_correlation=correlation_threshold,
                         pre_selection_number=pre_selection_number,
                         pre_selection_correlation=0,
                         verbose=False)
    for i in range(assets.asset_nb):
        for j in range(assets.asset_nb):
            if i != j:
                assert assets.corr[i, j] < correlation_threshold
    assert pre_selection_number <= len(assets.names) <= random_selection


def test_load_train_test_assets():
    prices = load_prices(file=TEST_PRICES_PATH)

    train_period = (dt.date(2018, 1, 1), dt.date(2019, 1, 1))
    test_period = (dt.date(2019, 1, 1), dt.date(2020, 1, 1))

    train_assets, test_assets = load_train_test_assets(prices=prices,
                                                       train_period=train_period,
                                                       test_period=test_period,
                                                       random_selection=15,
                                                       pre_selection_number=10,
                                                       pre_selection_correlation=0,
                                                       verbose=False)

    assert set(train_assets.names) == set(test_assets.names)
    assert (train_assets.start_date, train_assets.end_date) == train_period
    assert (test_assets.start_date, test_assets.end_date) == test_period


def test_load_assets_speed():
    prices = load_prices(file=TEST_PRICES_PATH)
    import time

    correlation_threshold = 0.99
    random_selection = 200
    pre_selection_number = 10

    s = time.time()
    assets = load_assets(prices=prices,
                         start_date=dt.date(2018, 1, 1),
                         end_date=dt.date(2019, 1, 1),
                         random_selection=random_selection,
                         removal_correlation=correlation_threshold,
                         pre_selection_number=pre_selection_number,
                         pre_selection_correlation=0,
                         verbose=False)
    e = time.time()
    print((e - s) * 1000)

    # 6531-4092
