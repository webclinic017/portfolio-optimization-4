import datetime as dt
import numpy as np

from portfolio_optimization.assets import *


def test_assets():
    assets = Assets(start_date=dt.date(2019, 1, 1))
    new_names = [assets.names[i] for i in np.random.choice(assets.asset_nb, 30, replace=False)]
    assets = Assets(start_date=dt.date(2019, 1, 1), names_to_keep=new_names)
    k = assets.asset_nb / 2
    new_assets_names = pre_selection(assets=assets, k=k)
    assert len(new_assets_names) >= k
    assert len(new_assets_names) < assets.asset_nb
