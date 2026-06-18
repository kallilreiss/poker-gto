"""
Strategy engine for GTO poker analysis.
Provides hand evaluation and preloaded GTO lookup tables.
"""

from enum import Enum
from itertools import combinations


RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER)}

SUIT_CHARS = set("cdhs")


class HandCategory(Enum):
    HIGH_CARD = "High Card"
    ONE_PAIR = "One Pair"
    TWO_PAIR = "Two Pair"
    THREE_OF_A_KIND = "Three of a Kind"
    STRAIGHT = "Straight"
    FLUSH = "Flush"
    FULL_HOUSE = "Full House"
    FOUR_OF_A_KIND = "Four of a Kind"
    STRAIGHT_FLUSH = "Straight Flush"
    GUTSHOT = "Gutshot Draw"
    OPEN_ENDED_STRAIGHT_DRAW = "Open-Ended Straight Draw"
    FLUSH_DRAW = "Flush Draw"
    OVERCARDS = "Overcards"


def parse_card(card: str) -> tuple[str, str]:
    """
    Parses a card string into (rank, suit).

    Args:
        card: Card string like "7c", "Ah", "Td".

    Returns:
        Tuple of (rank_char, suit_char).
    """
    if len(card) < 2:
        return ("?", "?")
    return (card[0].upper(), card[1].lower())


def categorize_hand(hole_cards: list[str] | str, board: list[str]) -> HandCategory:
    """
    Evaluates the best 5-card hand from hole cards + board and returns its category.

    Also checks for draws if no made hand is strong.

    Args:
        hole_cards: List of 2 card strings, or a single string like "7c4c".
        board: List of 3-5 community card strings.

    Returns:
        HandCategory enum value.
    """
    if isinstance(hole_cards, str):
        # Parse "7c4c" → ["7c", "4c"]
        if len(hole_cards) == 4:
            hole_cards = [hole_cards[:2], hole_cards[2:]]
        else:
            hole_cards = [hole_cards]

    all_cards = list(hole_cards) + list(board)
    parsed = [parse_card(c) for c in all_cards if len(c) >= 2]

    if len(parsed) < 2:
        return HandCategory.HIGH_CARD

    best = _best_hand_category(parsed)

    if best == HandCategory.HIGH_CARD and len(parsed) >= 4:
        draw = _detect_draw(hole_cards, board, parsed)
        if draw:
            return draw

    return best


def _best_hand_category(cards: list[tuple[str, str]]) -> HandCategory:
    """
    Finds the best 5-card hand among all combinations from the given cards.

    Args:
        cards: List of (rank, suit) tuples.

    Returns:
        Best HandCategory.
    """
    if len(cards) < 5:
        return _evaluate_short(cards)

    best = HandCategory.HIGH_CARD
    for combo in combinations(cards, 5):
        cat = _evaluate_five(list(combo))
        if _category_rank(cat) > _category_rank(best):
            best = cat

    return best


def _evaluate_five(cards: list[tuple[str, str]]) -> HandCategory:
    """
    Evaluates exactly 5 cards and returns hand category.

    Args:
        cards: Exactly 5 (rank, suit) tuples.

    Returns:
        HandCategory.
    """
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    values = sorted([RANK_VALUE.get(r, 0) for r in ranks], reverse=True)

    is_flush = len(set(suits)) == 1
    is_straight = _is_straight(values)

    if is_flush and is_straight:
        return HandCategory.STRAIGHT_FLUSH

    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    freq = sorted(counts.values(), reverse=True)

    if freq[0] == 4:
        return HandCategory.FOUR_OF_A_KIND
    if freq[0] == 3 and freq[1] == 2:
        return HandCategory.FULL_HOUSE
    if is_flush:
        return HandCategory.FLUSH
    if is_straight:
        return HandCategory.STRAIGHT
    if freq[0] == 3:
        return HandCategory.THREE_OF_A_KIND
    if freq[0] == 2 and freq[1] == 2:
        return HandCategory.TWO_PAIR
    if freq[0] == 2:
        return HandCategory.ONE_PAIR

    return HandCategory.HIGH_CARD


def _evaluate_short(cards: list[tuple[str, str]]) -> HandCategory:
    """Evaluates fewer than 5 cards."""
    counts = {}
    for r, _ in cards:
        counts[r] = counts.get(r, 0) + 1
    freq = sorted(counts.values(), reverse=True)
    if freq[0] >= 2:
        return HandCategory.ONE_PAIR
    return HandCategory.HIGH_CARD


def _is_straight(values: list[int]) -> bool:
    """Checks if sorted values form a straight."""
    if len(values) < 5:
        return False
    unique = sorted(set(values), reverse=True)
    if len(unique) < 5:
        return False
    if unique[0] - unique[4] == 4:
        return True
    # Wheel: A-2-3-4-5
    if set([12, 0, 1, 2, 3]).issubset(set(unique)):
        return True
    return False


def _detect_draw(
    hole_cards: list[str],
    board: list[str],
    all_parsed: list[tuple[str, str]],
) -> HandCategory | None:
    """
    Detects drawing hands (flush draw, OESD, gutshot).

    Args:
        hole_cards: Hero's two hole cards.
        board: Community cards.
        all_parsed: All (rank, suit) tuples.

    Returns:
        Draw HandCategory or None.
    """
    # Flush draw: 4 cards of same suit
    suit_counts: dict[str, int] = {}
    for _, suit in all_parsed:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    if any(v == 4 for v in suit_counts.values()):
        return HandCategory.FLUSH_DRAW

    # Straight draws
    values = sorted(set(RANK_VALUE.get(r, 0) for r, _ in all_parsed))

    # OESD: 4 consecutive cards
    for i in range(len(values) - 3):
        window = values[i:i + 4]
        if window[-1] - window[0] == 3 and len(window) == 4:
            return HandCategory.OPEN_ENDED_STRAIGHT_DRAW

    # Gutshot: 4 cards spanning 5 with one gap
    for i in range(len(values) - 3):
        window = values[i:i + 4]
        if window[-1] - window[0] == 4 and len(window) == 4:
            return HandCategory.GUTSHOT

    # Two overcards to the board
    if board:
        board_values = [RANK_VALUE.get(parse_card(c)[0], 0) for c in board]
        max_board = max(board_values)
        hole_values = [RANK_VALUE.get(parse_card(c)[0], 0) for c in hole_cards]
        overcards = sum(1 for v in hole_values if v > max_board)
        if overcards >= 2:
            return HandCategory.OVERCARDS

    return None


def _category_rank(cat: HandCategory) -> int:
    """Returns numeric rank for comparing HandCategory values."""
    order = [
        HandCategory.HIGH_CARD,
        HandCategory.OVERCARDS,
        HandCategory.GUTSHOT,
        HandCategory.OPEN_ENDED_STRAIGHT_DRAW,
        HandCategory.FLUSH_DRAW,
        HandCategory.ONE_PAIR,
        HandCategory.TWO_PAIR,
        HandCategory.THREE_OF_A_KIND,
        HandCategory.STRAIGHT,
        HandCategory.FLUSH,
        HandCategory.FULL_HOUSE,
        HandCategory.FOUR_OF_A_KIND,
        HandCategory.STRAIGHT_FLUSH,
    ]
    try:
        return order.index(cat)
    except ValueError:
        return 0


def build_preflop_table() -> dict:
    """
    Builds a preflop GTO lookup table.
    Covers common spots: RFI (Raise First In) by position.

    Returns:
        Dict mapping spot keys to GTO recommendation dicts.
    """
    table = {}

    # BTN RFI ranges
    btn_raise_hands = {
        "AAs", "KKs", "QQs", "JJs", "TTs", "99s", "88s", "77s", "66s", "55s",
        "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs", "ATo", "A9s", "A8s",
        "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KQo", "KJs", "KJo", "KTs", "KTo", "K9s",
        "QJs", "QJo", "QTs", "QTo", "Q9s",
        "JTs", "JTo", "J9s",
        "T9s", "T9o", "98s", "87s", "76s", "65s", "54s",
    }

    for hand in btn_raise_hands:
        table[f"BTN_{hand}"] = {
            "action": "RAISE",
            "frequencies": {"RAISE": 100},
            "bet_size": "2.5bb",
            "justification": f"BTN RFI com {hand}: abertura padrão em posição.",
        }

    # BB vs BTN (check or 3bet)
    bb_3bet_hands = {"AAs", "KKs", "QQs", "AKs", "AKo"}
    bb_defend_hands = {
        "JJs", "TTs", "99s", "AQs", "AQo", "AJs", "KQs", "JTs", "T9s", "98s",
    }

    for hand in bb_3bet_hands:
        table[f"BB_{hand}"] = {
            "action": "RAISE",
            "frequencies": {"RAISE": 90, "CALL": 10},
            "bet_size": "9-10bb",
            "justification": f"BB vs BTN 3-bet com {hand}: mão premium.",
        }

    for hand in bb_defend_hands:
        table[f"BB_{hand}"] = {
            "action": "CALL",
            "frequencies": {"CALL": 70, "RAISE": 30},
            "justification": f"BB vs BTN defend com {hand}: boa equity para defender.",
        }

    # CO RFI
    co_raise_hands = {
        "AAs", "KKs", "QQs", "JJs", "TTs", "99s", "88s", "77s",
        "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs",
        "KQs", "KQo", "KJs", "KJo", "QJs", "QJo", "JTs",
        "T9s", "98s", "87s", "76s",
    }

    for hand in co_raise_hands:
        table[f"CO_{hand}"] = {
            "action": "RAISE",
            "frequencies": {"RAISE": 100},
            "bet_size": "2.5bb",
            "justification": f"CO RFI com {hand}.",
        }

    return table


def build_postflop_table() -> dict:
    """
    Builds a postflop GTO lookup table covering common spots.

    Returns:
        Dict mapping spot keys to GTO recommendation dicts.
    """
    table = {}

    categories = [cat.value for cat in HandCategory]

    # IP (In Position) spots
    for cat in HandCategory:
        for street in ("Flop", "Turn", "River"):
            key = f"IP_{street}_{cat.value}"
            table[key] = _ip_postflop_strategy(cat, street)

    # OOP spots
    for cat in HandCategory:
        for street in ("Flop", "Turn", "River"):
            key = f"OOP_{street}_{cat.value}"
            table[key] = _oop_postflop_strategy(cat, street)

    return table


def _ip_postflop_strategy(cat: HandCategory, street: str) -> dict:
    """Returns IP postflop strategy for given hand category and street."""
    if cat in (HandCategory.STRAIGHT_FLUSH, HandCategory.FOUR_OF_A_KIND):
        return {
            "action": "BET",
            "frequencies": {"BET": 90, "CHECK": 10},
            "bet_size": "75%",
            "justification": f"Mão nuts ({cat.value}) em posição. Bet grande para valor máximo.",
        }
    if cat in (HandCategory.FULL_HOUSE, HandCategory.FLUSH, HandCategory.STRAIGHT):
        return {
            "action": "BET",
            "frequencies": {"BET": 80, "CHECK": 20},
            "bet_size": "66%",
            "justification": f"Mão muito forte ({cat.value}). Bet 2/3 pot para extrair valor.",
        }
    if cat in (HandCategory.THREE_OF_A_KIND, HandCategory.TWO_PAIR):
        return {
            "action": "BET",
            "frequencies": {"BET": 65, "CHECK": 35},
            "bet_size": "50%",
            "justification": f"{cat.value}: bet para proteção e valor em posição.",
        }
    if cat == HandCategory.ONE_PAIR:
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 55, "BET": 30, "FOLD": 15},
            "bet_size": "33%",
            "justification": "Par médio em posição. Mix: check para pot control, bet 1/3 para proteção.",
        }
    if cat == HandCategory.FLUSH_DRAW:
        return {
            "action": "BET",
            "frequencies": {"BET": 55, "CHECK": 45},
            "bet_size": "50%",
            "justification": "Flush draw em posição. Semi-bluff com equity e fold equity combinados.",
        }
    if cat == HandCategory.OPEN_ENDED_STRAIGHT_DRAW:
        return {
            "action": "BET",
            "frequencies": {"BET": 50, "CHECK": 50},
            "bet_size": "33%",
            "justification": "OESD em posição. Semi-bluff pequeno para pressão com equity.",
        }
    if cat == HandCategory.GUTSHOT:
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 65, "BET": 35},
            "bet_size": "33%",
            "justification": "Gutshot: menos equity que OESD. Frequência de bet reduzida.",
        }
    if cat == HandCategory.OVERCARDS:
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 70, "BET": 30},
            "bet_size": "33%",
            "justification": "Duas sobrecards em posição. Check com opção de float.",
        }
    return {
        "action": "CHECK",
        "frequencies": {"CHECK": 80, "BET": 20},
        "bet_size": "33%",
        "justification": "Mão fraca em posição. Check predominante, bet ocasional como bluff.",
    }


def _oop_postflop_strategy(cat: HandCategory, street: str) -> dict:
    """Returns OOP postflop strategy for given hand category and street."""
    if cat in (HandCategory.STRAIGHT_FLUSH, HandCategory.FOUR_OF_A_KIND):
        return {
            "action": "BET",
            "frequencies": {"BET": 85, "CHECK": 15},
            "bet_size": "75%",
            "justification": f"Mão nuts ({cat.value}) fora de posição. Donk bet ou check-raise.",
        }
    if cat in (HandCategory.FULL_HOUSE, HandCategory.FLUSH, HandCategory.STRAIGHT):
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 60, "BET": 40},
            "bet_size": "66%",
            "justification": f"Mão muito forte ({cat.value}) OOP. Check para induzir ou donk 2/3 pot.",
        }
    if cat in (HandCategory.THREE_OF_A_KIND, HandCategory.TWO_PAIR):
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 65, "BET": 35},
            "bet_size": "50%",
            "justification": f"{cat.value} OOP. Majoritariamente check para check-raise vs bet.",
        }
    if cat == HandCategory.ONE_PAIR:
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 75, "BET": 15, "FOLD": 10},
            "bet_size": "33%",
            "justification": "Par médio OOP. Check é a ação dominante para controle de pote.",
        }
    if cat in (HandCategory.FLUSH_DRAW, HandCategory.OPEN_ENDED_STRAIGHT_DRAW):
        return {
            "action": "CHECK",
            "frequencies": {"CHECK": 60, "BET": 40},
            "bet_size": "50%",
            "justification": f"{cat.value} OOP. Check-raise é opção forte; semi-bluff balanceado.",
        }
    return {
        "action": "CHECK",
        "frequencies": {"CHECK": 82, "BET": 18},
        "bet_size": "33%",
        "justification": "Mão fraca OOP. Check predominante. Fold a apostas grandes.",
    }
