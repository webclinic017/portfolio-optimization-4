import datetime as dt
import numpy as np

from portfolio_optimization.assets import *


def test_assets():
    assets = Assets(date_from=dt.date(2019, 1, 1))
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
    assets= Assets(date_from=dt.date(2019, 1, 1), names_to_keep=new_names)
    assert assets.asset_nb == len(new_names)
    assert assets.date_nb == len(assets.prices) - 1