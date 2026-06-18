"""
GTO lookup engine.
Queries preloaded strategy tables and returns action frequencies for a given spot.
"""

import json
import os
from pathlib import Path

from gto_engine.strategy_engine import (
    build_preflop_table,
    build_postflop_table,
    HandCategory,
    categorize_hand,
)


_PREFLOP_TABLE: dict | None = None
_POSTFLOP_TABLE: dict | None = None


def get_gto_action(
    hero_cards: list[str],
    board: list[str],
    position: str,
    street: str,
    pot_bb: float,
    hero_stack: float,
    villain_stacks: list[float],
) -> dict:
    """
    Returns GTO-recommended action and frequencies for the given spot.

    Args:
        hero_cards: Hero's hole cards, e.g. ["7c", "4c"].
        board: Community cards, e.g. ["4h", "Js", "Kd"].
        position: Hero's position, e.g. "BB".
        street: Current street, e.g. "Flop".
        pot_bb: Pot size in BB.
        hero_stack: Hero's remaining stack in BB.
        villain_stacks: List of villain stack sizes in BB.

    Returns:
        Dict with keys: action, frequencies, justification.
    """
    global _PREFLOP_TABLE, _POSTFLOP_TABLE

    if _PREFLOP_TABLE is None:
        _PREFLOP_TABLE = build_preflop_table()

    if _POSTFLOP_TABLE is None:
        _POSTFLOP_TABLE = build_postflop_table()

    hand_str = "".join(hero_cards)

    if street == "Preflop":
        return _lookup_preflop(hand_str, position, hero_stack, _PREFLOP_TABLE)

    return _lookup_postflop(
        hand_str, board, position, street, pot_bb, hero_stack, _POSTFLOP_TABLE
    )


def _lookup_preflop(
    hand: str, position: str, stack: float, table: dict
) -> dict:
    """
    Looks up preflop GTO action.

    Args:
        hand: Two-card hand string e.g. "AKs", "77".
        position: Table position.
        stack: Hero's stack in BB.
        table: Preflop GTO table dict.

    Returns:
        GTO recommendation dict.
    """
    hand_key = _normalize_hand_key(hand)
    pos_key = position.upper()

    # Try exact match
    spot_key = f"{pos_key}_{hand_key}"
    if spot_key in table:
        return table[spot_key]

    # Try position category
    pos_category = _position_category(pos_key)
    cat_key = f"{pos_category}_{hand_key}"
    if cat_key in table:
        return table[cat_key]

    # Evaluate hand strength and use generic strategy
    strength = _hand_strength_preflop(hand_key)
    return _generic_preflop_action(strength, position, stack)


def _lookup_postflop(
    hand: str,
    board: list[str],
    position: str,
    street: str,
    pot_bb: float,
    stack: float,
    table: dict,
) -> dict:
    """
    Looks up postflop GTO action.

    Args:
        hand: Hero's hole cards string.
        board: Community cards list.
        position: Table position.
        street: Current street.
        pot_bb: Pot size in BB.
        stack: Hero's stack in BB.
        table: Postflop GTO table dict.

    Returns:
        GTO recommendation dict.
    """
    category = categorize_hand(hand, board)
    spr = stack / pot_bb if pot_bb > 0 else 10
    pos_category = _position_category(position)

    spot_key = f"{pos_category}_{street}_{category.value}"
    if spot_key in table:
        entry = table[spot_key].copy()
        entry = _adjust_for_spr(entry, spr)
        return entry

    # Generic postflop by hand category and position
    return _generic_postflop_action(category, position, street, pot_bb, stack)


def _normalize_hand_key(hand: str) -> str:
    """
    Normalizes a hand string to a canonical key like "AKs", "72o", "TT".

    Args:
        hand: Raw hand string (e.g. "7c4c", "AhKd").

    Returns:
        Canonical hand key.
    """
    if len(hand) < 2:
        return hand.upper()

    ranks = []
    suits = []
    i = 0
    while i < len(hand):
        ch = hand[i]
        if ch.upper() in "AKQJT98765432":
            ranks.append(ch.upper())
            if i + 1 < len(hand) and hand[i + 1].lower() in "cdhs":
                suits.append(hand[i + 1].lower())
                i += 2
            else:
                i += 1
        else:
            i += 1

    if len(ranks) < 2:
        return hand.upper()

    r1, r2 = ranks[0], ranks[1]
    rank_order = "AKQJT98765432"

    if rank_order.index(r1) > rank_order.index(r2):
        r1, r2 = r2, r1
        if suits:
            suits = suits[::-1]

    if r1 == r2:
        return f"{r1}{r2}"

    suited = len(suits) == 2 and suits[0] == suits[1]
    return f"{r1}{r2}{'s' if suited else 'o'}"


def _position_category(position: str) -> str:
    """Groups positions into IP (in position) or OOP (out of position) categories."""
    ip_positions = {"BTN", "CO", "HJ"}
    if position in ip_positions:
        return "IP"
    return "OOP"


def _hand_strength_preflop(hand_key: str) -> str:
    """Classifies preflop hand strength as premium/strong/medium/weak/trash."""
    premium = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo"}
    strong = {"99", "88", "77", "AQs", "AQo", "AJs", "KQs", "ATs"}
    medium = {"66", "55", "44", "AJo", "KQo", "KJs", "QJs", "JTs"}

    if hand_key in premium:
        return "premium"
    if hand_key in strong:
        return "strong"
    if hand_key in medium:
        return "medium"

    # Suited connectors and Broadway
    if hand_key.endswith("s") and len(hand_key) == 3:
        r1, r2 = hand_key[0], hand_key[1]
        rank_order = "AKQJT98765432"
        if abs(rank_order.index(r1) - rank_order.index(r2)) == 1:
            return "medium"

    return "weak"


def _generic_preflop_action(strength: str, position: str, stack: float) -> dict:
    """Returns a generic preflop GTO action based on hand strength."""
    actions = {
        "premium": {
            "action": "RAISE",
            "frequencies": {"RAISE": 95, "CALL": 5},
            "justification": (
                "Mão premium. Raise para construir o pote e ganhar valor com a melhor mão."
            ),
        },
        "strong": {
            "action": "RAISE",
            "frequencies": {"RAISE": 70, "CALL": 20, "FOLD": 10},
            "justification": (
                "Mão forte. Raise como abertura na maioria das posições. "
                "Manter range equilibrado."
            ),
        },
        "medium": {
            "action": "CALL" if position in ("BTN", "CO") else "RAISE",
            "frequencies": {"RAISE": 40, "CALL": 35, "FOLD": 25},
            "justification": (
                "Mão média. Playability depende da posição e das ações anteriores."
            ),
        },
        "weak": {
            "action": "FOLD",
            "frequencies": {"FOLD": 70, "CALL": 20, "RAISE": 10},
            "justification": (
                "Mão fraca. Fold é a ação padrão fora de posição. "
                "Abrir apenas em BTN/CO com stack adequado."
            ),
        },
    }
    return actions.get(strength, actions["weak"])


def _generic_postflop_action(
    category: "HandCategory",
    position: str,
    street: str,
    pot_bb: float,
    stack: float,
) -> dict:
    """Returns generic postflop action based on hand category."""
    from gto_engine.strategy_engine import HandCategory

    spr = stack / pot_bb if pot_bb > 0 else 10
    is_ip = position in ("BTN", "CO", "HJ")

    if category in (HandCategory.STRAIGHT_FLUSH, HandCategory.FOUR_OF_A_KIND,
                    HandCategory.FULL_HOUSE, HandCategory.FLUSH, HandCategory.STRAIGHT):
        return {
            "action": "BET",
            "frequencies": {"BET": 85, "CHECK": 15},
            "bet_size": "75%",
            "justification": (
                f"Mão muito forte ({category.value}). Aposta para extrair valor máximo. "
                f"SPR={spr:.1f}, construir pote agressivamente."
            ),
        }

    if category in (HandCategory.THREE_OF_A_KIND, HandCategory.TWO_PAIR):
        return {
            "action": "BET" if is_ip else "CHECK",
            "frequencies": {"BET": 65 if is_ip else 30, "CHECK": 35 if is_ip else 70},
            "bet_size": "50%",
            "justification": (
                f"Mão forte ({category.value}). "
                + ("Aposta para proteger e extrair valor em posição." if is_ip
                   else "Check fora de posição para controlar o pote e manter range.")
            ),
        }

    if category == HandCategory.ONE_PAIR:
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 60, "BET": 25, "FOLD": 15},
            "bet_size": "33%",
            "justification": (
                f"Par médio. Estratégia mista: check para pot control, "
                f"bet pequeno para proteção. SPR={spr:.1f}."
            ),
        }

    if category in (HandCategory.FLUSH_DRAW, HandCategory.OPEN_ENDED_STRAIGHT_DRAW):
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 55, "BET": 45},
            "bet_size": "50%",
            "justification": (
                f"Draw forte ({category.value}). Semi-bluff com equity. "
                f"Frequência de aposta aumenta em posição."
            ),
        }

    # High card / weak
    return {
        "action": "CHECK",
        "frequencies": {"CHECK": 80, "BET": 20},
        "bet_size": "33%",
        "justification": (
            f"Mão fraca ({category.value}). Check é a ação predominante. "
            "Pequenas apostas ocasionais como bluff para equilibrar o range."
        ),
    }


def _adjust_for_spr(entry: dict, spr: float) -> dict:
    """
    Adjusts action recommendation based on stack-to-pot ratio.

    Args:
        entry: Base GTO entry dict.
        spr: Stack-to-pot ratio.

    Returns:
        Adjusted entry dict.
    """
    if spr < 2:
        # Short stack: prefer all-in or fold
        entry["justification"] += f" SPR={spr:.1f}: pilha curta, considere all-in."
    elif spr > 15:
        entry["justification"] += f" SPR={spr:.1f}: pilha funda, jogo mais cauteloso."

    return entry
