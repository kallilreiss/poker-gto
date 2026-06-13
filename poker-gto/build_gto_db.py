"""
build_gto_db.py
Cria o banco de dados SQLite com tabelas GTO completas para 6max e 9max.
Execute este arquivo UMA VEZ antes de subir no Streamlit.
"""

import sqlite3
import json
import os

DB_PATH = "gto_poker.db"

# ─── POSIÇÕES ────────────────────────────────────────────────────────────────
POSITIONS_6MAX = ["BTN", "CO", "HJ", "MP", "SB", "BB"]
POSITIONS_9MAX = ["BTN", "CO", "HJ", "MP2", "MP1", "UTG1", "UTG", "SB", "BB"]

# ─── MÃOS PREFLOP (169 combos canônicos) ────────────────────────────────────
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

def all_hands():
    hands = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i < j:
                hands.append(f"{r1}{r2}s")  # suited
            elif i > j:
                hands.append(f"{r2}{r1}o")  # offsuit
            else:
                hands.append(f"{r1}{r2}")   # pair
    return list(dict.fromkeys(hands))

ALL_HANDS = all_hands()

# ─── RANGES GTO PREFLOP (RFI = Raise First In) ───────────────────────────────
# Valores: R=Raise, C=Call, F=Fold, L=Limp
# Baseado em solvers populares (GTO+, PioSOLVER ranges publicados)

def build_rfi_ranges():
    """Ranges de abertura por posição — 6max e 9max."""
    ranges = {}

    # ── 6MAX RFI ──────────────────────────────────────────────────────────────
    ranges["6max"] = {
        "BTN": {  # ~45% das mãos
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R", "77o": "R", "66o": "R", "55o": "R",
            "44o": "R", "33o": "R", "22o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A8s": "R", "A7s": "R", "A6s": "R", "A5s": "R", "A4s": "R",
            "A3s": "R", "A2s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R", "A9o": "R",
            "A8o": "R", "A7o": "R",
            "KQs": "R", "KJs": "R", "KTs": "R", "K9s": "R", "K8s": "R",
            "K7s": "R", "K6s": "R", "K5s": "R",
            "KQo": "R", "KJo": "R", "KTo": "R", "K9o": "R",
            "QJs": "R", "QTs": "R", "Q9s": "R", "Q8s": "R", "Q7s": "R",
            "QJo": "R", "QTo": "R", "Q9o": "R",
            "JTs": "R", "J9s": "R", "J8s": "R", "J7s": "R",
            "JTo": "R", "J9o": "R",
            "T9s": "R", "T8s": "R", "T7s": "R",
            "T9o": "R", "T8o": "R",
            "98s": "R", "97s": "R", "96s": "R",
            "98o": "R",
            "87s": "R", "86s": "R", "85s": "R",
            "87o": "R",
            "76s": "R", "75s": "R", "74s": "R",
            "65s": "R", "64s": "R",
            "54s": "R", "53s": "R",
            "43s": "R",
        },
        "CO": {  # ~35%
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R", "77o": "R", "66o": "R", "55o": "R",
            "44o": "R", "33o": "R", "22o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A8s": "R", "A7s": "R", "A6s": "R", "A5s": "R", "A4s": "R",
            "A3s": "R", "A2s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R", "A9o": "R",
            "A8o": "R",
            "KQs": "R", "KJs": "R", "KTs": "R", "K9s": "R", "K8s": "R",
            "K7s": "R", "K6s": "R",
            "KQo": "R", "KJo": "R", "KTo": "R", "K9o": "R",
            "QJs": "R", "QTs": "R", "Q9s": "R", "Q8s": "R",
            "QJo": "R", "QTo": "R",
            "JTs": "R", "J9s": "R", "J8s": "R",
            "JTo": "R", "J9o": "R",
            "T9s": "R", "T8s": "R", "T7s": "R",
            "T9o": "R",
            "98s": "R", "97s": "R",
            "87s": "R", "86s": "R",
            "76s": "R", "75s": "R",
            "65s": "R", "64s": "R",
            "54s": "R",
        },
        "HJ": {  # ~28%
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R", "77o": "R", "66o": "R", "55o": "R",
            "44o": "R", "33o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A8s": "R", "A7s": "R", "A6s": "R", "A5s": "R", "A4s": "R",
            "A3s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R", "A9o": "R",
            "KQs": "R", "KJs": "R", "KTs": "R", "K9s": "R", "K8s": "R",
            "K7s": "R",
            "KQo": "R", "KJo": "R", "KTo": "R",
            "QJs": "R", "QTs": "R", "Q9s": "R",
            "QJo": "R", "QTo": "R",
            "JTs": "R", "J9s": "R", "J8s": "R",
            "JTo": "R",
            "T9s": "R", "T8s": "R",
            "98s": "R", "97s": "R",
            "87s": "R", "86s": "R",
            "76s": "R",
            "65s": "R",
            "54s": "R",
        },
        "MP": {  # ~22%
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R", "77o": "R", "66o": "R", "55o": "R",
            "44o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A8s": "R", "A7s": "R", "A5s": "R", "A4s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R",
            "KQs": "R", "KJs": "R", "KTs": "R", "K9s": "R",
            "KQo": "R", "KJo": "R",
            "QJs": "R", "QTs": "R", "Q9s": "R",
            "QJo": "R",
            "JTs": "R", "J9s": "R",
            "T9s": "R", "T8s": "R",
            "98s": "R",
            "87s": "R",
            "76s": "R",
            "65s": "R",
            "54s": "R",
        },
        "SB": {  # ~40% (limp ou raise)
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R", "77o": "R", "66o": "R", "55o": "R",
            "44o": "R", "33o": "R", "22o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A8s": "R", "A7s": "R", "A6s": "R", "A5s": "R", "A4s": "R",
            "A3s": "R", "A2s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R", "A9o": "R",
            "A8o": "L", "A7o": "L", "A6o": "L",
            "KQs": "R", "KJs": "R", "KTs": "R", "K9s": "R", "K8s": "R",
            "K7s": "L", "K6s": "L", "K5s": "L",
            "KQo": "R", "KJo": "R", "KTo": "R", "K9o": "L",
            "QJs": "R", "QTs": "R", "Q9s": "R", "Q8s": "L",
            "QJo": "R", "QTo": "L",
            "JTs": "R", "J9s": "R", "J8s": "L",
            "JTo": "L",
            "T9s": "R", "T8s": "L",
            "98s": "R", "97s": "L",
            "87s": "L", "86s": "L",
            "76s": "L", "75s": "L",
            "65s": "L", "64s": "L",
            "54s": "L", "53s": "L",
            "43s": "L",
        },
        "BB": {  # BB defende vs raise
            # Preenchido na tabela de defesa (vs_raise)
        },
    }

    # ── 9MAX RFI ──────────────────────────────────────────────────────────────
    ranges["9max"] = {
        "BTN": ranges["6max"]["BTN"],  # BTN idêntico
        "CO": ranges["6max"]["CO"],
        "HJ": ranges["6max"]["HJ"],
        "MP2": {  # ~18%
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R", "77o": "R", "66o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A8s": "R", "A5s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R",
            "KQs": "R", "KJs": "R", "KTs": "R",
            "KQo": "R", "KJo": "R",
            "QJs": "R", "QTs": "R",
            "QJo": "R",
            "JTs": "R",
            "T9s": "R",
            "98s": "R",
            "87s": "R",
            "76s": "R",
            "65s": "R",
        },
        "MP1": {  # ~15%
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R", "77o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A5s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R",
            "KQs": "R", "KJs": "R", "KTs": "R",
            "KQo": "R", "KJo": "R",
            "QJs": "R", "QTs": "R",
            "JTs": "R",
            "T9s": "R",
            "98s": "R",
            "87s": "R",
            "76s": "R",
        },
        "UTG1": {  # ~13%
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A5s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R", "ATo": "R",
            "KQs": "R", "KJs": "R", "KTs": "R",
            "KQo": "R", "KJo": "R",
            "QJs": "R", "QTs": "R",
            "JTs": "R",
            "T9s": "R",
            "98s": "R",
            "87s": "R",
        },
        "UTG": {  # ~12%
            "AAo": "R", "KKo": "R", "QQo": "R", "JJo": "R", "TTo": "R",
            "99o": "R", "88o": "R",
            "AKs": "R", "AQs": "R", "AJs": "R", "ATs": "R", "A9s": "R",
            "A5s": "R",
            "AKo": "R", "AQo": "R", "AJo": "R",
            "KQs": "R", "KJs": "R", "KTs": "R",
            "KQo": "R",
            "QJs": "R", "QTs": "R",
            "JTs": "R",
            "T9s": "R",
            "98s": "R",
        },
        "SB": ranges["6max"]["SB"],
        "BB": {},
    }

    return ranges


def build_vs_raise_ranges():
    """Ranges de defesa do BB vs raise de cada posição."""
    # BB vs BTN raise (mais amplo)
    bb_vs_btn = {
        "AAo": "3B", "KKo": "3B", "QQo": "3B", "JJo": "3B", "TTo": "3B",
        "99o": "C", "88o": "C", "77o": "C", "66o": "C", "55o": "C",
        "44o": "C", "33o": "C", "22o": "C",
        "AKs": "3B", "AQs": "3B", "AJs": "3B", "ATs": "3B", "A9s": "C",
        "A8s": "C", "A7s": "C", "A6s": "C", "A5s": "3B", "A4s": "C",
        "A3s": "C", "A2s": "C",
        "AKo": "3B", "AQo": "3B", "AJo": "3B", "ATo": "C", "A9o": "C",
        "A8o": "C", "A7o": "F", "A6o": "F",
        "KQs": "3B", "KJs": "C", "KTs": "C", "K9s": "C", "K8s": "C",
        "K7s": "C", "K6s": "C", "K5s": "C",
        "KQo": "C", "KJo": "C", "KTo": "C", "K9o": "C", "K8o": "F",
        "QJs": "C", "QTs": "C", "Q9s": "C", "Q8s": "C",
        "QJo": "C", "QTo": "C", "Q9o": "C",
        "JTs": "C", "J9s": "C", "J8s": "C",
        "JTo": "C", "J9o": "C",
        "T9s": "C", "T8s": "C", "T7s": "C",
        "T9o": "C",
        "98s": "C", "97s": "C", "96s": "C",
        "98o": "C",
        "87s": "C", "86s": "C", "85s": "C",
        "76s": "C", "75s": "C",
        "65s": "C", "64s": "C",
        "54s": "C", "53s": "C",
        "43s": "C",
    }

    # BB vs UTG raise (mais restrito)
    bb_vs_utg = {
        "AAo": "3B", "KKo": "3B", "QQo": "3B", "JJo": "3B", "TTo": "3B",
        "99o": "C", "88o": "C", "77o": "C", "66o": "F", "55o": "F",
        "AKs": "3B", "AQs": "3B", "AJs": "C", "ATs": "C", "A9s": "C",
        "A8s": "C", "A5s": "C", "A4s": "C",
        "AKo": "3B", "AQo": "3B", "AJo": "C", "ATo": "C",
        "KQs": "C", "KJs": "C", "KTs": "C", "K9s": "C",
        "KQo": "C", "KJo": "C",
        "QJs": "C", "QTs": "C",
        "JTs": "C",
        "T9s": "C",
        "98s": "C",
        "87s": "C",
        "76s": "C",
    }

    return {"BTN": bb_vs_btn, "CO": bb_vs_btn, "HJ": bb_vs_utg,
            "MP": bb_vs_utg, "MP2": bb_vs_utg, "MP1": bb_vs_utg,
            "UTG1": bb_vs_utg, "UTG": bb_vs_utg}


def build_3bet_ranges():
    """Ranges de 3bet por posição vs raiser."""
    return {
        "BTN_vs_CO": {
            "AAo": "3B", "KKo": "3B", "QQo": "3B", "JJo": "3B", "TTo": "3B",
            "99o": "C", "88o": "C",
            "AKs": "3B", "AQs": "3B", "AJs": "3B", "ATs": "C",
            "A5s": "3B", "A4s": "3B",
            "AKo": "3B", "AQo": "3B", "AJo": "C",
            "KQs": "3B", "KJs": "C", "KTs": "C",
            "KQo": "C",
            "QJs": "C",
            "JTs": "C",
        },
        "BTN_vs_UTG": {
            "AAo": "3B", "KKo": "3B", "QQo": "3B", "JJo": "C", "TTo": "C",
            "AKs": "3B", "AQs": "3B", "AJs": "C",
            "A5s": "3B",
            "AKo": "3B", "AQo": "C",
            "KQs": "C",
        },
    }


def build_postflop_heuristics():
    """Heurísticas postflop: tipo de mão → ação recomendada."""
    return [
        # (hand_category, board_texture, position, stack_to_pot, action, reason)
        ("straight_flush", "any", "any", "any", "BET_BIG", "Mão nutted — extraia valor máximo"),
        ("quads", "any", "any", "any", "BET_BIG", "Quads — slowplay ou value bet grande"),
        ("full_house", "any", "any", "any", "BET_BIG", "Full house — aposta por valor"),
        ("flush", "paired_board", "any", "any", "BET_MED", "Flush em board pareado — cuidado com full house"),
        ("flush", "unpaired_board", "any", "any", "BET_BIG", "Flush em board não pareado — extraia valor"),
        ("straight", "flush_possible", "any", "any", "BET_MED", "Straight com flush possível — bet por valor mas cauteloso"),
        ("straight", "no_flush", "any", "any", "BET_BIG", "Straight em board seco — aposta grande"),
        ("trips", "any", "IP", "any", "BET_MED", "Trips em posição — bet por valor"),
        ("trips", "any", "OOP", "any", "BET_MED", "Trips fora de posição — bet ou check-raise"),
        ("two_pair", "monotone", "any", "any", "CHECK_CALL", "Dois pares em board monotone — cuidado"),
        ("two_pair", "paired_board", "any", "any", "BET_MED", "Dois pares em board pareado — bet por valor"),
        ("two_pair", "unpaired_board", "any", "any", "BET_BIG", "Dois pares em board não pareado — aposta por valor"),
        ("overpair", "dry", "IP", "high", "BET_MED", "Overpair em board seco — value bet"),
        ("overpair", "wet", "any", "any", "CHECK_CALL", "Overpair em board molhado — pot control"),
        ("top_pair_top_kicker", "dry", "IP", "high", "BET_MED", "TPTK seco — aposta por valor"),
        ("top_pair_top_kicker", "wet", "any", "any", "BET_SMALL", "TPTK molhado — bet pequeno ou check"),
        ("top_pair_weak_kicker", "dry", "IP", "any", "BET_SMALL", "Top pair kicker fraco — bet pequeno"),
        ("top_pair_weak_kicker", "wet", "any", "any", "CHECK_CALL", "Top pair kicker fraco molhado — controle"),
        ("middle_pair", "any", "IP", "any", "CHECK_CALL", "Par do meio — controle de pote"),
        ("bottom_pair", "any", "any", "any", "CHECK_FOLD", "Par de baixo — dificilmente continua"),
        ("flush_draw", "any", "IP", "any", "BET_SMALL", "Flush draw — semi-bluff pequeno"),
        ("flush_draw", "any", "OOP", "any", "CHECK_RAISE", "Flush draw OOP — check-raise semi-bluff"),
        ("oesd", "any", "IP", "any", "BET_SMALL", "OESD — semi-bluff"),
        ("gutshot", "any", "any", "any", "CHECK_CALL", "Gutshot — pot odds dependente"),
        ("overcards", "dry", "any", "any", "CHECK_FOLD", "Overcards em board seco — fold geralmente"),
        ("air", "any", "any", "any", "CHECK_FOLD", "Sem equity — fold"),
    ]


def create_database():
    """Cria e popula o banco de dados SQLite."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── Tabela de ranges RFI ───────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS preflop_rfi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_type TEXT NOT NULL,
            position TEXT NOT NULL,
            hand TEXT NOT NULL,
            action TEXT NOT NULL,
            UNIQUE(game_type, position, hand)
        )
    """)

    rfi_ranges = build_rfi_ranges()
    for game_type, positions in rfi_ranges.items():
        for position, hands in positions.items():
            for hand, action in hands.items():
                c.execute("""
                    INSERT OR IGNORE INTO preflop_rfi (game_type, position, hand, action)
                    VALUES (?, ?, ?, ?)
                """, (game_type, position, hand, action))

    # ── Tabela de defesa BB vs raise ───────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS preflop_bb_defense (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raiser_position TEXT NOT NULL,
            hand TEXT NOT NULL,
            action TEXT NOT NULL,
            UNIQUE(raiser_position, hand)
        )
    """)

    vs_raise = build_vs_raise_ranges()
    for raiser_pos, hands in vs_raise.items():
        for hand, action in hands.items():
            c.execute("""
                INSERT OR IGNORE INTO preflop_bb_defense (raiser_position, hand, action)
                VALUES (?, ?, ?)
            """, (raiser_pos, hand, action))

    # ── Tabela de 3bet ranges ──────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS preflop_3bet (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario TEXT NOT NULL,
            hand TEXT NOT NULL,
            action TEXT NOT NULL,
            UNIQUE(scenario, hand)
        )
    """)

    three_bet = build_3bet_ranges()
    for scenario, hands in three_bet.items():
        for hand, action in hands.items():
            c.execute("""
                INSERT OR IGNORE INTO preflop_3bet (scenario, hand, action)
                VALUES (?, ?, ?)
            """, (scenario, hand, action))

    # ── Tabela de heurísticas postflop ────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS postflop_heuristics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_category TEXT NOT NULL,
            board_texture TEXT NOT NULL,
            position_type TEXT NOT NULL,
            stack_to_pot TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL
        )
    """)

    for row in build_postflop_heuristics():
        c.execute("""
            INSERT INTO postflop_heuristics
            (hand_category, board_texture, position_type, stack_to_pot, action, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, row)

    # ── Tabela de pot odds / fold equity ──────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS pot_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bet_to_call REAL NOT NULL,
            pot_size REAL NOT NULL,
            required_equity REAL NOT NULL,
            decision TEXT NOT NULL
        )
    """)

    # Preenche com valores calculados
    for bet_pct in [25, 33, 50, 67, 75, 100, 125, 150, 200]:
        pot = 100.0
        call = pot * bet_pct / 100
        total = pot + call
        equity = call / (pot + call * 2)
        decision = "CALL_OR_BLUFF" if equity < 0.40 else ("MARGINAL" if equity < 0.45 else "FOLD_WITHOUT_EQUITY")
        c.execute("""
            INSERT INTO pot_odds (bet_to_call, pot_size, required_equity, decision)
            VALUES (?, ?, ?, ?)
        """, (round(call, 2), pot, round(equity, 4), decision))

    # ── Tabela de posições por tipo de mesa ───────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS table_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_type TEXT NOT NULL,
            seat_number INTEGER NOT NULL,
            position_name TEXT NOT NULL,
            position_type TEXT NOT NULL
        )
    """)

    pos_data_6max = [
        ("6max", 1, "BTN", "late"),
        ("6max", 2, "SB", "blind"),
        ("6max", 3, "BB", "blind"),
        ("6max", 4, "MP", "early"),
        ("6max", 5, "HJ", "middle"),
        ("6max", 6, "CO", "late"),
    ]
    pos_data_9max = [
        ("9max", 1, "BTN", "late"),
        ("9max", 2, "SB", "blind"),
        ("9max", 3, "BB", "blind"),
        ("9max", 4, "UTG", "early"),
        ("9max", 5, "UTG1", "early"),
        ("9max", 6, "MP1", "middle"),
        ("9max", 7, "MP2", "middle"),
        ("9max", 8, "HJ", "middle"),
        ("9max", 9, "CO", "late"),
    ]

    for row in pos_data_6max + pos_data_9max:
        c.execute("""
            INSERT INTO table_positions (game_type, seat_number, position_name, position_type)
            VALUES (?, ?, ?, ?)
        """, row)

    conn.commit()
    conn.close()

    # Estatísticas
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    print("✅ Banco de dados criado com sucesso!")
    for table in ["preflop_rfi", "preflop_bb_defense", "preflop_3bet",
                  "postflop_heuristics", "pot_odds", "table_positions"]:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"   {table}: {count} registros")
    conn.close()
    print(f"\n📁 Arquivo: {DB_PATH}")


if __name__ == "__main__":
    create_database()
