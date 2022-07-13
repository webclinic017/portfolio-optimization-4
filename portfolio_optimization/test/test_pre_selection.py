import datetime as dt
import numpy as np

from portfolio_optimization.assets import *
from portfolio_optimization.loader import *
from portfolio_optimization.paths import *
from portfolio_optimization.bloomberg import *


def test_assets():
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
    new_assets_names = pre_selection(assets=assets, k=k)
    assert len(new_assets_names) >= k
    assert len(new_assets_names) < assets.asset_nb
