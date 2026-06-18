"""
Unit tests for Poker Vision GTO.
Tests cover: hand evaluation, GTO engine, OCR parsers, position logic.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Strategy engine ────────────────────────────────────────────────────────────
from gto_engine.strategy_engine import (
    categorize_hand,
    HandCategory,
    parse_card,
    _is_straight,
    RANK_VALUE,
)
from gto_engine.lookup import _normalize_hand_key


class TestParseCard:
    def test_simple(self):
        assert parse_card("7c") == ("7", "c")

    def test_ace_hearts(self):
        assert parse_card("Ah") == ("A", "h")

    def test_ten(self):
        assert parse_card("Td") == ("T", "d")

    def test_short_string(self):
        r, s = parse_card("A")
        # Single char without suit returns safe defaults
        assert r in ("A", "?")


class TestNormalizeHandKey:
    def test_suited(self):
        assert _normalize_hand_key("AhKh") == "AKs"

    def test_offsuit(self):
        assert _normalize_hand_key("AhKd") == "AKo"

    def test_pair(self):
        assert _normalize_hand_key("7c7h") == "77"

    def test_order_maintained(self):
        # Lower card first in string → still normalized high-to-low
        assert _normalize_hand_key("2hAc") == "A2o"


class TestIsStraight:
    def test_broadway(self):
        vals = [RANK_VALUE[r] for r in ["A", "K", "Q", "J", "T"]]
        assert _is_straight(sorted(vals, reverse=True))

    def test_wheel(self):
        vals = [RANK_VALUE[r] for r in ["A", "2", "3", "4", "5"]]
        assert _is_straight(sorted(vals, reverse=True))

    def test_not_straight(self):
        vals = [RANK_VALUE[r] for r in ["A", "K", "Q", "J", "9"]]
        assert not _is_straight(sorted(vals, reverse=True))


class TestCategorizeHand:
    def test_pair_on_flop(self):
        result = categorize_hand(["7c", "4c"], ["4h", "Js", "Kd"])
        assert result == HandCategory.ONE_PAIR

    def test_two_pair(self):
        result = categorize_hand(["7c", "4c"], ["7h", "4s", "Kd"])
        assert result == HandCategory.TWO_PAIR

    def test_trips(self):
        result = categorize_hand(["7c", "7d"], ["7h", "4s", "Kd"])
        assert result == HandCategory.THREE_OF_A_KIND

    def test_flush_draw(self):
        result = categorize_hand(["7c", "4c"], ["2c", "9c", "Kd"])
        assert result == HandCategory.FLUSH_DRAW

    def test_oesd(self):
        result = categorize_hand(["8c", "7d"], ["5h", "6s", "Kd"])
        assert result == HandCategory.OPEN_ENDED_STRAIGHT_DRAW

    def test_straight(self):
        result = categorize_hand(["8c", "7d"], ["5h", "6s", "9c"])
        assert result == HandCategory.STRAIGHT

    def test_preflop_high_card(self):
        result = categorize_hand(["7c", "2d"], [])
        assert result == HandCategory.HIGH_CARD

    def test_full_house(self):
        result = categorize_hand(["Kc", "Kd"], ["Kh", "4s", "4d"])
        assert result == HandCategory.FULL_HOUSE

    def test_string_hand_input(self):
        result = categorize_hand("7c4c", ["4h", "Js", "Kd"])
        assert result == HandCategory.ONE_PAIR


# ── GTO Lookup ─────────────────────────────────────────────────────────────────
from gto_engine.lookup import get_gto_action


class TestGTOLookup:
    def test_returns_dict(self):
        result = get_gto_action(
            hero_cards=["7c", "4c"],
            board=["4h", "Js", "Kd"],
            position="BB",
            street="Flop",
            pot_bb=5.67,
            hero_stack=37.0,
            villain_stacks=[74.7],
        )
        assert isinstance(result, dict)
        assert "action" in result
        assert "frequencies" in result
        assert "justification" in result

    def test_action_is_string(self):
        result = get_gto_action(
            hero_cards=["Ac", "Kd"],
            board=[],
            position="BTN",
            street="Preflop",
            pot_bb=1.5,
            hero_stack=100.0,
            villain_stacks=[100.0],
        )
        assert isinstance(result["action"], str)

    def test_frequencies_sum_to_100(self):
        result = get_gto_action(
            hero_cards=["7c", "4c"],
            board=["4h", "Js", "Kd"],
            position="BB",
            street="Flop",
            pot_bb=5.67,
            hero_stack=37.0,
            villain_stacks=[],
        )
        total = sum(result["frequencies"].values())
        assert 95 <= total <= 105  # Allow rounding

    def test_preflop_premium_raises(self):
        result = get_gto_action(
            hero_cards=["Ac", "Ad"],
            board=[],
            position="BTN",
            street="Preflop",
            pot_bb=1.5,
            hero_stack=100.0,
            villain_stacks=[100.0],
        )
        assert result["action"] == "RAISE"

    def test_weak_hand_oop_checks(self):
        result = get_gto_action(
            hero_cards=["7c", "2d"],
            board=["Ah", "Ks", "Qd"],
            position="BB",
            street="Flop",
            pot_bb=4.0,
            hero_stack=50.0,
            villain_stacks=[50.0],
        )
        assert result["action"] == "CHECK"


# ── OCR parsers ────────────────────────────────────────────────────────────────
from table_detector.detect_pot import _parse_pot
from table_detector.detect_stacks import _parse_bb_value


class TestParsePot:
    def test_pote_label(self):
        assert _parse_pot("Pote: 5,67 BB") == pytest.approx(5.67, rel=1e-3)

    def test_pot_english(self):
        assert _parse_pot("Pot 10 BB") == pytest.approx(10.0)

    def test_bare_number(self):
        assert _parse_pot("23.4") == pytest.approx(23.4)

    def test_empty(self):
        assert _parse_pot("") is None

    def test_with_noise(self):
        assert _parse_pot("Total pot: 163.1 BB some garbage") == pytest.approx(163.1)


class TestParseStack:
    def test_simple(self):
        assert _parse_bb_value("74.7 BB") == pytest.approx(74.7)

    def test_comma_decimal(self):
        assert _parse_bb_value("138,7 BB") == pytest.approx(138.7)

    def test_integer(self):
        assert _parse_bb_value("99 BB") == pytest.approx(99.0)

    def test_no_bb_label(self):
        # Falls back to number pattern
        result = _parse_bb_value("454.2")
        assert result == pytest.approx(454.2)

    def test_empty_default(self):
        assert _parse_bb_value("", default=0.0) == 0.0


# ── Position ───────────────────────────────────────────────────────────────────
from table_detector.detect_positions import _find_nearest_seat, SEAT_COORDINATES_9MAX


class TestFindNearestSeat:
    def test_hero_seat(self):
        # Bottom-center of a 1440x900 image → should be seat 0 (hero)
        idx = _find_nearest_seat((720, 738), SEAT_COORDINATES_9MAX, 1440, 900)
        assert idx == 0

    def test_top_center(self):
        # Top-center → seat 4 or 5 (top-center-right / top-center-left)
        idx = _find_nearest_seat((720, 72), SEAT_COORDINATES_9MAX, 1440, 900)
        assert idx in (4, 5)

    def test_right_side(self):
        # Far right → seat 2
        idx = _find_nearest_seat((1325, 378), SEAT_COORDINATES_9MAX, 1440, 900)
        assert idx == 2


# ── Street detection ──────────────────────────────────────────────────────────
from table_detector.detect_street import detect_street
from unittest.mock import patch


class TestDetectStreet:
    def test_preflop(self):
        img = np.zeros((900, 1440, 3), dtype=np.uint8)
        with patch("table_detector.detect_street.detect_board", return_value=[]):
            assert detect_street(img) == "Preflop"

    def test_flop(self):
        img = np.zeros((900, 1440, 3), dtype=np.uint8)
        with patch("table_detector.detect_street.detect_board", return_value=["4h", "Js", "Kd"]):
            assert detect_street(img) == "Flop"

    def test_turn(self):
        img = np.zeros((900, 1440, 3), dtype=np.uint8)
        with patch("table_detector.detect_street.detect_board", return_value=["4h", "Js", "Kd", "2c"]):
            assert detect_street(img) == "Turn"

    def test_river(self):
        img = np.zeros((900, 1440, 3), dtype=np.uint8)
        with patch("table_detector.detect_street.detect_board",
                   return_value=["4h", "Js", "Kd", "2c", "9s"]):
            assert detect_street(img) == "River"
