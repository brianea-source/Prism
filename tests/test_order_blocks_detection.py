"""Phase 6.B — OrderBlockDetector.detect / update_states / queries."""

import pandas as pd
import pytest

from prism.signal.order_blocks import OrderBlock, OrderBlockDetector, OrderBlockState


def _eur_row(o: float, h: float, l: float, c: float) -> dict:
    return {"open": o, "high": h, "low": l, "close": c}


def test_detect_bullish_ob_after_displacement():
    # Bar 1: bearish zone; bar 2: impulse clears zone high by >10 pips (EUR pip=0.0001)
    rows = [
        _eur_row(1.1060, 1.1065, 1.1055, 1.1062),
        _eur_row(1.1060, 1.1062, 1.1040, 1.1042),  # bearish opposing
        _eur_row(1.1045, 1.1075, 1.1041, 1.1070),  # bullish displacement
        _eur_row(1.1070, 1.1072, 1.1068, 1.1071),  # post — for update_states
    ]
    df = pd.DataFrame(rows)
    det = OrderBlockDetector("EURUSD", "H4")
    new = det.detect(df, min_displacement_pips=10.0)
    assert len(new) >= 1
    ob = new[0]
    assert ob.direction == "BULLISH"
    assert ob.state == OrderBlockState.OB_FRESH
    assert ob.formed_bar == 1
    assert ob.high == 1.1062 and ob.low == 1.1040
    assert ob.activation_bar == 3
    assert ob.displacement_size >= 10.0


def test_detect_bearish_ob_after_displacement():
    rows = [
        _eur_row(1.1000, 1.1010, 1.0995, 1.1005),
        _eur_row(1.1000, 1.1010, 1.0998, 1.1008),  # bullish opposing
        _eur_row(1.1005, 1.1009, 1.0880, 1.0890),  # bearish displacement
        _eur_row(1.0890, 1.0892, 1.0888, 1.0891),
    ]
    df = pd.DataFrame(rows)
    det = OrderBlockDetector("EURUSD", "H4")
    new = det.detect(df, min_displacement_pips=10.0)
    bearish = [b for b in new if b.direction == "BEARISH"]
    assert len(bearish) >= 1
    ob = bearish[0]
    assert ob.formed_bar == 1
    assert ob.low == 1.0998
    assert ob.activation_bar == 3


def test_min_displacement_pips_honored():
    # Two bearish bars then a weak bullish impulse <10 pips extension past zone low —
    # no bullish OB. No lone bullish bar before bar 2 so bearish OB path also stays empty.
    rows = [
        _eur_row(1.1060, 1.1062, 1.1040, 1.1042),
        _eur_row(1.1045, 1.1048, 1.1041, 1.1043),
        _eur_row(1.1043, 1.1049, 1.1042, 1.1048),
    ]
    df = pd.DataFrame(rows)
    det = OrderBlockDetector("EURUSD", "H4")
    new = det.detect(df, min_displacement_pips=10.0)
    assert new == []


def test_update_states_idempotent_same_df():
    rows = [
        _eur_row(1.1060, 1.1065, 1.1055, 1.1062),
        _eur_row(1.1060, 1.1062, 1.1040, 1.1042),
        _eur_row(1.1045, 1.1075, 1.1041, 1.1070),
        _eur_row(1.1070, 1.1100, 1.1045, 1.1095),  # overlap zone [1.1040,1.1062]: low 1.1045
    ]
    df = pd.DataFrame(rows)
    det = OrderBlockDetector("EURUSD", "H4")
    det.detect(df, min_displacement_pips=10.0)
    det.update_states(df)
    ob = next(b for b in det.blocks if b.direction == "BULLISH")
    assert ob.state == OrderBlockState.OB_TESTED
    s1 = ob.state
    det.update_states(df)
    assert ob.state == s1


def test_get_active_blocks_filters_consumed_and_age():
    det = OrderBlockDetector("EURUSD", "H4")
    b1 = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.11,
        low=1.10,
        midpoint=1.105,
        formed_at="t",
        formed_bar=0,
        displacement_size=12.0,
        state=OrderBlockState.OB_FRESH,
        age_bars=5,
    )
    b2 = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BEARISH",
        high=1.12,
        low=1.11,
        midpoint=1.115,
        formed_at="t",
        formed_bar=1,
        displacement_size=10.0,
        state=OrderBlockState.CONSUMED,
        age_bars=0,
    )
    b3 = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.09,
        low=1.08,
        midpoint=1.085,
        formed_at="t",
        formed_bar=2,
        displacement_size=15.0,
        state=OrderBlockState.OB_FRESH,
        age_bars=60,
    )
    det.blocks.extend([b1, b2, b3])
    act = det.get_active_blocks(max_age_bars=50)
    assert len(act) == 1 and act[0] is b1


def test_get_nearest_ob_matches_effective_direction_for_rb():
    det = OrderBlockDetector("EURUSD", "H4")
    ob = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.10,
        low=1.09,
        midpoint=1.095,
        formed_at="t",
        formed_bar=5,
        displacement_size=20.0,
        state=OrderBlockState.RB_FRESH,
    )
    det.blocks.append(ob)
    # effective BEARISH → SHORT
    nearest = det.get_nearest_ob(1.10, "SHORT")
    assert nearest is ob
    assert det.get_nearest_ob(1.10, "LONG") is None


def test_get_nearest_ob_long_returns_bullish_ob():
    det = OrderBlockDetector("EURUSD", "H4")
    far = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.20,
        low=1.19,
        midpoint=1.195,
        formed_at="t",
        formed_bar=0,
        displacement_size=10.0,
    )
    near = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.11,
        low=1.10,
        midpoint=1.105,
        formed_at="t",
        formed_bar=1,
        displacement_size=10.0,
    )
    det.blocks.extend([far, near])
    assert det.get_nearest_ob(1.10, "LONG") is near


def test_distance_to_ob_calculation():
    det = OrderBlockDetector("EURUSD", "H4")
    ob = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.11,
        low=1.09,
        midpoint=1.10,
        formed_at="t",
        formed_bar=0,
        displacement_size=10.0,
    )
    det.blocks.append(ob)
    # price 1.101 → 10 pips from midpoint 1.10
    d = det.distance_to_ob(1.101, "LONG")
    assert d == pytest.approx(10.0)


def test_htf_priority_filter_4h_overrides_1h():
    det = OrderBlockDetector("EURUSD", "H4")
    pip = det._pip_size()
    h1 = OrderBlock(
        instrument="EURUSD",
        timeframe="H1",
        direction="BULLISH",
        high=1.10 + 3 * pip,
        low=1.10,
        midpoint=1.10 + 1.5 * pip,
        formed_at="t",
        formed_bar=1,
        displacement_size=10.0,
    )
    h4 = OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.10 + 3 * pip,
        low=1.10,
        midpoint=1.10 + 2 * pip,  # within 5 pips of h1.midpoint
        formed_at="t",
        formed_bar=2,
        displacement_size=12.0,
    )
    out = det.htf_priority_filter([h1, h4])
    assert len(out) == 1
    assert out[0].timeframe == "H4"


def test_detect_requires_ohlc_columns():
    det = OrderBlockDetector("EURUSD", "H4")
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="open"):
        det.detect(df)


def test_detect_empty_or_short_df():
    det = OrderBlockDetector("EURUSD", "H4")
    assert det.detect(pd.DataFrame(), min_displacement_pips=10.0) == []
    df = pd.DataFrame([_eur_row(1, 1, 1, 1), _eur_row(1, 1, 1, 1)])
    assert det.detect(df, min_displacement_pips=10.0) == []


def test_catch_up_when_detect_after_update_states():
    # Process full df first, then detect — new block must replay processed tail
    rows = [
        _eur_row(1.1060, 1.1065, 1.1055, 1.1062),
        _eur_row(1.1060, 1.1062, 1.1040, 1.1042),
        _eur_row(1.1045, 1.1075, 1.1041, 1.1070),
        _eur_row(1.1070, 1.1100, 1.1045, 1.1095),
    ]
    df = pd.DataFrame(rows)
    det = OrderBlockDetector("EURUSD", "H4")
    det.update_states(df)
    assert det._cursor == 4
    det.detect(df, min_displacement_pips=10.0)
    ob = next(b for b in det.blocks if b.direction == "BULLISH")
    assert ob.state == OrderBlockState.OB_TESTED


def test_get_nearest_ob_invalid_direction_returns_none():
    det = OrderBlockDetector("EURUSD", "H4")
    det.blocks.append(
        OrderBlock(
            instrument="EURUSD",
            timeframe="H4",
            direction="BULLISH",
            high=1.1,
            low=1.09,
            midpoint=1.095,
            formed_at="t",
            formed_bar=0,
            displacement_size=10.0,
        )
    )
    assert det.get_nearest_ob(1.0, "DIAGONAL") is None
