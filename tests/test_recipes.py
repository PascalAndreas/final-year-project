from datetime import date
from pathlib import Path
from functools import partial
import sys
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from okx.store import OrderbookStore
from okx.recipes.forwards import prepare_pillars, build_forwards_pchip
from okx.recipes.options import prepare_options


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "okx"
MANIFEST_PATH = DATA_ROOT / "manifest.sqlite"
TEST_DATES = [date(2025, 8, 5)]


@pytest.fixture(scope="session")
def store() -> OrderbookStore:
    if not DATA_ROOT.exists():
        pytest.skip("OKX data directory not available; skip integration tests")
    return OrderbookStore(
        data_root=str(DATA_ROOT),
        manifest_path=str(MANIFEST_PATH),
    )


def test_prepare_pillars_real_data_sorted(store: OrderbookStore):
    lf = prepare_pillars(
        store=store,
        inst_family='BTC-USD',
        dates=TEST_DATES,
        binning='5m',
        drop_pillar_idx=1,
    )
    df = lf.collect()

    assert not df.is_empty()
    ordering_view = df.select(['timeMs', 'T'])
    assert ordering_view.frame_equal(ordering_view.sort(['timeMs', 'T']))

    counts = df.group_by('timeMs').count().sort('timeMs')
    # With drop_pillar_idx=1 we should still have at least 1 pillar per timestamp
    assert counts['count'].min() >= 1


def test_build_forwards_pchip_real_data(store: OrderbookStore):
    lf = build_forwards_pchip(
        store=store,
        dates=TEST_DATES,
        inst_family='BTC-USD',
        binning='5m',
        tau_ewma_minutes=5.0,
    )
    df = lf.collect()

    required_cols = {'timeMs', 'T', 'ln_F_bid', 'ln_F_ask', 'symbol'}
    assert required_cols.issubset(set(df.columns))
    assert df.height > 0


def test_prepare_options_with_real_data(store: OrderbookStore):
    recipe = partial(build_forwards_pchip, binning='5m', tau_ewma_minutes=5.0)

    df = prepare_options(
        store=store,
        inst_family='BTC-USD',
        dates=TEST_DATES,
        forwards_recipe=recipe,
        binning='5m',
    )

    assert not df.is_empty()
    assert {'F_fitted_bid', 'F_fitted_ask', 'moneyness'}.issubset(df.columns)
    assert df.filter(pl.col('opt_type') == 'C')['F_fitted_mid'].min() > 0
