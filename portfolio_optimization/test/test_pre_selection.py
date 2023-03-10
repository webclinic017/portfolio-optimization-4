import datetime as dt
import numpy as np
import time

from portfolio_optimization.assets import *
from portfolio_optimization.loader import *
from portfolio_optimization.paths import *
from portfolio_optimization.utils.tools import *


def test_pre_selection():
    prices = load_prices(file=TEST_PRICES_PATH)

    start_date = dt.date(2017, 1, 1)
    assets = Assets(prices=prices,
                    start_date=start_date,
                    verbose=False)
    new_names = [assets.names[i] for i in np.random.choice(assets.asset_nb, 10, replace=False)]
    assets = Assets(prices=prices,
                    start_date=start_date,
                    names_to_keep=new_names,
                    verbose=False)
    k = assets.asset_nb / 2
    s = time.time()
    new_assets_names = pre_selection(assets=assets, k=k, correlation_threshold=0)
    e = time.time()
    print((e - s))
    assert e - s < 3
    assert len(new_assets_names) >= k
    assert len(new_assets_names) < assets.asset_nb
