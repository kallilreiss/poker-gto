"""
hand_evaluator.py
Avalia força da mão no postflop e classifica board texture.
Zero dependências externas além de Python padrão.
"""

from itertools import combinations
from collections import Counter

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}
SUITS = ["s", "h", "d", "c"]


def parse_card(card_str):
    """Parseia '9h' → ('9', 'h')."""
    if not card_str or len(card_str) < 2:
        return None
    card_str = card_str.strip()
    suit = card_str[-1].lower()
    rank = card_str[:-1].upper().replace("10", "T")
    if rank in RANK_VALUES and suit in SUITS:
        return (rank, suit)
    return None


def hand_rank(cards):
    """
    Avalia a melhor mão de 5 cartas a partir de até 7 cartas.
    Retorna (rank_int, tiebreakers) onde rank_int:
    8=Straight Flush, 7=Quads, 6=Full House, 5=Flush,
    4=Straight, 3=Trips, 2=Two Pair, 1=One Pair, 0=High Card
    """
    parsed = [parse_card(c) for c in cards if c]
    parsed = [c for c in parsed if c]

    if len(parsed) < 2:
        return (0, [])

    best = None
    for combo in combinations(parsed, min(5, len(parsed))):
        score = evaluate_5card(list(combo))
        if best is None or score > best:
            best = score
    return best if best else (0, [])


def evaluate_5card(cards):
    """Avalia exatamente 5 cartas."""
    ranks = sorted([RANK_VALUES[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]

    is_flush = len(set(suits)) == 1
    is_straight = _is_straight(ranks)

    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    rank_groups = sorted(rank_counts.keys(),
                          key=lambda r: (rank_counts[r], r), reverse=True)

    if is_flush and is_straight:
        return (8, ranks)
    if counts[0] == 4:
        return (7, rank_groups)
    if counts[0] == 3 and counts[1] == 2:
        return (6, rank_groups)
    if is_flush:
        return (5, ranks)
    if is_straight:
        return (4, ranks)
    if counts[0] == 3:
        return (3, rank_groups)
    if counts[0] == 2 and counts[1] == 2:
        return (2, rank_groups)
    if counts[0] == 2:
        return (1, rank_groups)
    return (0, ranks)


def _is_straight(sorted_ranks):
    """Verifica se é sequência (incluindo A-2-3-4-5)."""
    if len(sorted_ranks) < 5:
        return False
    if sorted_ranks[0] - sorted_ranks[4] == 4 and len(set(sorted_ranks)) == 5:
        return True
    # Wheel: A-2-3-4-5
    if sorted_ranks == [12, 3, 2, 1, 0]:
        return True
    return False


class HandEvaluator:

    HAND_CATEGORIES = {
        8: "straight_flush",
        7: "quads",
        6: "full_house",
        5: "flush",
        4: "straight",
    }

    def __init__(self):
        pass

    def evaluate(self, hole_cards, board_cards):
        """
        Avalia a mão completa e retorna categoria detalhada.

        hole_cards: list de str, ex: ["Ah", "Ks"]
        board_cards: list de str, ex: ["2h", "7d", "Tc"]
        """
        if not hole_cards:
            return {"category": "unknown", "strength": 0, "description": "Sem cartas"}

        all_cards = hole_cards + board_cards
        score, tiebreakers = hand_rank(all_cards)

        # Categorias simples por score
        if score in self.HAND_CATEGORIES:
            cat = self.HAND_CATEGORIES[score]
            return {
                "category": cat,
                "strength": score,
                "description": self._describe(cat, tiebreakers),
                "score": score,
            }

        # Análise mais detalhada para pares e high cards
        return self._detailed_pair_analysis(hole_cards, board_cards, score, tiebreakers)

    def _detailed_pair_analysis(self, hole_cards, board_cards, score, tiebreakers):
        """Análise detalhada para pares, dois pares, trinca e high cards."""
        if not board_cards:
            # Preflop — avalia força relativa da mão
            return self._preflop_strength(hole_cards)

        board_parsed = [parse_card(c) for c in board_cards if c]
        board_parsed = [c for c in board_parsed if c]
        hole_parsed  = [parse_card(c) for c in hole_cards if c]
        hole_parsed  = [c for c in hole_parsed if c]

        if not board_parsed:
            return {"category": "unknown", "strength": 0}

        board_ranks = sorted([RANK_VALUES[c[0]] for c in board_parsed], reverse=True)
        hole_ranks  = [RANK_VALUES[c[0]] for c in hole_parsed]

        top_board = board_ranks[0] if board_ranks else 0

        if score == 3:  # Trips
            return {"category": "trips", "strength": 3,
                    "description": f"Trinca de {RANKS[tiebreakers[0]]}"}

        if score == 2:  # Two pair
            return {"category": "two_pair", "strength": 2,
                    "description": "Dois pares"}

        if score == 1:  # One pair
            return self._classify_one_pair(hole_ranks, board_ranks, top_board)

        # High card — draws ou overcards
        return self._classify_high_card(hole_parsed, board_parsed)

    def _classify_one_pair(self, hole_ranks, board_ranks, top_board):
        """Classifica tipo de par."""
        if not hole_ranks or not board_ranks:
            return {"category": "middle_pair", "strength": 1, "description": "Par"}

        top_hole = max(hole_ranks)

        # Pocket pair
        if len(hole_ranks) >= 2 and hole_ranks[0] == hole_ranks[1]:
            if hole_ranks[0] > top_board:
                return {"category": "overpair", "strength": 1.8,
                        "description": f"Overpair {RANKS[hole_ranks[0]]}{RANKS[hole_ranks[0]]}"}
            return {"category": "middle_pair", "strength": 1.5,
                    "description": "Pocket pair (underpair)"}

        # Par no board
        if top_hole == top_board:
            # Verifica kicker
            kicker = min(hole_ranks) if len(hole_ranks) > 1 else 0
            if kicker >= 10:  # J ou melhor
                return {"category": "top_pair_top_kicker", "strength": 1.7,
                        "description": f"TPTK — {RANKS[top_hole]} + kicker {RANKS[kicker]}"}
            elif kicker >= 7:
                return {"category": "top_pair_good_kicker", "strength": 1.6,
                        "description": f"Top pair kicker médio"}
            else:
                return {"category": "top_pair_weak_kicker", "strength": 1.5,
                        "description": f"Top pair kicker fraco"}

        if len(board_ranks) > 1 and top_hole == board_ranks[1]:
            return {"category": "middle_pair", "strength": 1.3,
                    "description": "Middle pair"}

        return {"category": "bottom_pair", "strength": 1.1,
                "description": "Bottom pair"}

    def _classify_high_card(self, hole_parsed, board_parsed):
        """Classifica draws e overcards."""
        all_parsed = hole_parsed + board_parsed
        hole_ranks = [RANK_VALUES[c[0]] for c in hole_parsed]
        board_ranks = sorted([RANK_VALUES[c[0]] for c in board_parsed], reverse=True)
        all_ranks   = [RANK_VALUES[c[0]] for c in all_parsed]
        all_suits   = [c[1] for c in all_parsed]
        board_suits  = [c[1] for c in board_parsed]

        top_board = board_ranks[0] if board_ranks else 0

        # Flush draw (4 cartas do mesmo naipe)
        suit_counts = Counter(all_suits)
        board_suit_counts = Counter(board_suits)
        for suit, count in suit_counts.items():
            if count == 4:
                # Verifica se hole cards contribuem
                hole_suits = [c[1] for c in hole_parsed]
                if suit in hole_suits:
                    return {"category": "flush_draw", "strength": 0.9,
                            "description": "Flush draw (4 para flush)"}

        # OESD (open-ended straight draw)
        if self._has_oesd(all_ranks):
            return {"category": "oesd", "strength": 0.8,
                    "description": "OESD — open-ended straight draw"}

        # Gutshot
        if self._has_gutshot(all_ranks):
            return {"category": "gutshot", "strength": 0.6,
                    "description": "Gutshot straight draw"}

        # Overcards
        overcards = sum(1 for r in hole_ranks if r > top_board)
        if overcards == 2:
            return {"category": "overcards", "strength": 0.5,
                    "description": "Dois overcards — sem par ainda"}
        if overcards == 1:
            return {"category": "overcard", "strength": 0.4,
                    "description": "Um overcard"}

        return {"category": "air", "strength": 0.1,
                "description": "Sem equity relevante"}

    def _has_oesd(self, ranks):
        """Verifica OESD em lista de ranks."""
        unique = sorted(set(ranks))
        for i in range(len(unique) - 3):
            window = unique[i:i+4]
            if window[-1] - window[0] == 3 and len(window) == 4:
                return True
        return False

    def _has_gutshot(self, ranks):
        """Verifica gutshot em lista de ranks."""
        unique = sorted(set(ranks))
        for i in range(len(unique) - 3):
            window = unique[i:i+4]
            if window[-1] - window[0] == 4 and len(window) == 4:
                return True
        return False

    def _preflop_strength(self, hole_cards):
        """Avalia força preflop."""
        parsed = [parse_card(c) for c in hole_cards if c]
        parsed = [c for c in parsed if c]

        if len(parsed) < 2:
            return {"category": "unknown", "strength": 0}

        r1 = RANK_VALUES[parsed[0][0]]
        r2 = RANK_VALUES[parsed[1][0]]
        suited = parsed[0][1] == parsed[1][1]

        top = max(r1, r2)
        bot = min(r1, r2)
        is_pair = r1 == r2

        if is_pair and top >= 10:
            return {"category": "premium_pair", "strength": 9, "description": "Par premium"}
        if is_pair:
            return {"category": "pocket_pair", "strength": 7, "description": "Par no bolso"}
        if top == 12 and bot >= 11:
            return {"category": "premium", "strength": 9, "description": "Mão premium"}
        if top >= 11 and suited:
            return {"category": "broadway_suited", "strength": 7, "description": "Broadway suited"}
        if top >= 11:
            return {"category": "broadway", "strength": 6, "description": "Broadway offsuit"}
        if suited and top - bot <= 2:
            return {"category": "suited_connector", "strength": 5, "description": "Suited connector"}
        if suited:
            return {"category": "suited", "strength": 4, "description": "Suited"}

        return {"category": "weak", "strength": 2, "description": "Mão fraca"}

    def classify_board_texture(self, board_cards):
        """
        Classifica a textura do board.
        Retorna: dry, wet, monotone, paired_board, two_tone, rainbow
        """
        if not board_cards:
            return "none"

        parsed = [parse_card(c) for c in board_cards if c]
        parsed = [c for c in parsed if c]

        if len(parsed) < 3:
            return "preflop"

        ranks = [RANK_VALUES[c[0]] for c in parsed]
        suits = [c[1] for c in parsed]

        suit_counts = Counter(suits)
        rank_counts = Counter(ranks)

        max_suit = max(suit_counts.values())
        max_rank = max(rank_counts.values())

        # Paired board
        is_paired = max_rank >= 2
        # Monotone
        is_monotone = max_suit == len(parsed)
        # Two-tone (2 do mesmo naipe no flop)
        is_two_tone = max_suit == 2 and len(parsed) == 3

        # Conectividade
        sorted_ranks = sorted(set(ranks), reverse=True)
        gaps = [sorted_ranks[i] - sorted_ranks[i+1]
                for i in range(len(sorted_ranks)-1)] if len(sorted_ranks) > 1 else [99]
        is_connected = all(g <= 2 for g in gaps)
        is_dry = not is_connected and not is_two_tone and not is_monotone

        if is_monotone:
            return "monotone"
        if is_paired and is_connected:
            return "wet"
        if is_paired:
            return "paired_board"
        if is_monotone or is_two_tone:
            if is_connected:
                return "wet"
            return "flush_possible"
        if is_dry:
            return "dry"
        if is_connected:
            return "wet"

        return "unpaired_board"

    def get_outs(self, hole_cards, board_cards):
        """
        Calcula outs para draws.
        Retorna número de outs e equity aproximada.
        """
        if not hole_cards or not board_cards:
            return {"outs": 0, "equity": 0}

        parsed_hole  = [parse_card(c) for c in hole_cards if c]
        parsed_board = [parse_card(c) for c in board_cards if c]
        parsed_hole  = [c for c in parsed_hole if c]
        parsed_board = [c for c in parsed_board if c]

        all_parsed = parsed_hole + parsed_board
        known_cards = len(all_parsed)
        cards_to_come = 2 if len(parsed_board) == 3 else 1

        # Avalia mão atual
        current_eval = self.evaluate(hole_cards, board_cards)
        current_score = current_eval.get("strength", 0)

        # Outs por tipo de draw
        outs = 0
        draw_type = "none"

        all_ranks = [RANK_VALUES[c[0]] for c in all_parsed]
        all_suits  = [c[1] for c in all_parsed]
        hole_suits = [c[1] for c in parsed_hole]
        hole_ranks = [RANK_VALUES[c[0]] for c in parsed_hole]

        # Flush draw
        suit_counts = Counter(all_suits)
        for suit, count in suit_counts.items():
            if count == 4 and suit in hole_suits:
                outs = max(outs, 9)
                draw_type = "flush_draw"

        # Straight draw
        unique_ranks = sorted(set(all_ranks))
        for start in range(len(unique_ranks) - 3):
            window = unique_ranks[start:start+4]
            if window[-1] - window[0] == 3:
                outs = max(outs, 8)
                draw_type = "oesd"
            elif window[-1] - window[0] == 4:
                outs = max(outs, 4)
                if draw_type != "oesd":
                    draw_type = "gutshot"

        # Overcards
        top_board_rank = max([RANK_VALUES[c[0]] for c in parsed_board]) if parsed_board else 0
        overcards = sum(1 for r in hole_ranks if r > top_board_rank)
        if overcards == 2 and outs == 0:
            outs = 6
            draw_type = "two_overcards"

        # Fórmula de outs (regra de 4 e 2)
        if cards_to_come == 2:
            equity_pct = min(outs * 4, 100)
        else:
            equity_pct = min(outs * 2, 100)

        return {
            "outs": outs,
            "equity_pct": equity_pct,
            "draw_type": draw_type,
            "cards_to_come": cards_to_come,
        }

    def _describe(self, category, tiebreakers):
        names = {
            "straight_flush": "Straight Flush",
            "quads": "Quadra",
            "full_house": "Full House",
            "flush": "Flush",
            "straight": "Sequência",
        }
        base = names.get(category, category)
        if tiebreakers:
            top_rank = RANKS[tiebreakers[0]] if tiebreakers[0] < len(RANKS) else ""
            return f"{base} de {top_rank}" if top_rank else base
        return base
