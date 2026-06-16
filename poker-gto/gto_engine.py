"""
gto_engine.py
Motor de consulta GTO — busca decisões no banco de dados SQLite.
Zero IA — 100% lookup de tabelas.
"""

import sqlite3
import os

DB_PATH = "gto_poker.db"

RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

# Mapeamento de ações para texto amigável
ACTION_LABELS = {
    "R":   ("🟢 RAISE / OPEN", "Abra o pote com raise"),
    "C":   ("🔵 CALL",          "Pague a aposta"),
    "F":   ("🔴 FOLD",          "Descarte a mão"),
    "L":   ("🟡 LIMP",          "Entre sem raise (limp)"),
    "3B":  ("🔥 3-BET",         "Faça um re-raise"),
    "BET_BIG":    ("🟢 BET GRANDE (75-100% pot)", "Aposte por valor máximo"),
    "BET_MED":    ("🟢 BET MÉDIO (50-67% pot)",   "Aposte por valor"),
    "BET_SMALL":  ("🟡 BET PEQUENO (25-33% pot)", "Aposta de valor ou bluff"),
    "CHECK_CALL": ("🔵 CHECK / CALL",              "Controle de pote, chame apostas razoáveis"),
    "CHECK_RAISE":("🔥 CHECK-RAISE",               "Deixe o oponente apostar, então reraize"),
    "CHECK_FOLD": ("🔴 CHECK / FOLD",              "Sem equity suficiente"),
}

STREET_LABELS = {
    0: "Preflop",
    3: "Flop",
    4: "Turn",
    5: "River",
}


class GTOEngine:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._check_db()

    def _check_db(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Banco de dados '{self.db_path}' não encontrado.\n"
                "Execute: python build_gto_db.py"
            )

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # ─── PREFLOP ─────────────────────────────────────────────────────────────

    def get_preflop_action(self, hand, position, game_type, situation="rfi",
                            raiser_pos=None):
        """
        Retorna a ação GTO preflop.

        hand: str — ex: "AKs", "QJo", "TTo"
        position: str — ex: "BTN", "CO", "BB"
        game_type: str — "6max" ou "9max"
        situation: str — "rfi" (raise first in), "bb_defense", "3bet"
        raiser_pos: str — posição do raiser (quando situation != rfi)
        """
        conn = self._connect()
        c = conn.cursor()
        action = None
        source = None

        try:
            if situation == "rfi":
                c.execute("""
                    SELECT action FROM preflop_rfi
                    WHERE game_type = ? AND position = ? AND hand = ?
                """, (game_type, position, hand))
                row = c.fetchone()
                action = row[0] if row else "F"
                source = "RFI Range"

            elif situation == "bb_defense" and raiser_pos:
                c.execute("""
                    SELECT action FROM preflop_bb_defense
                    WHERE raiser_position = ? AND hand = ?
                """, (raiser_pos, hand))
                row = c.fetchone()
                action = row[0] if row else "F"
                source = f"BB Defense vs {raiser_pos}"

            elif situation == "3bet" and raiser_pos:
                scenario = f"{position}_vs_{raiser_pos}"
                c.execute("""
                    SELECT action FROM preflop_3bet
                    WHERE scenario = ? AND hand = ?
                """, (scenario, hand))
                row = c.fetchone()
                if not row:
                    # Tenta cenário genérico
                    c.execute("""
                        SELECT action FROM preflop_3bet
                        WHERE scenario LIKE ? AND hand = ?
                    """, (f"%vs_{raiser_pos}", hand))
                    row = c.fetchone()
                action = row[0] if row else "F"
                source = f"3-Bet Range ({position} vs {raiser_pos})"

        finally:
            conn.close()

        label, desc = ACTION_LABELS.get(action, (action, ""))
        return {
            "action": action,
            "label": label,
            "description": desc,
            "source": source,
            "hand": hand,
            "position": position,
        }

    def get_preflop_range_stats(self, position, game_type):
        """Retorna estatísticas do range de uma posição."""
        conn = self._connect()
        c = conn.cursor()
        try:
            c.execute("""
                SELECT action, COUNT(*) as cnt
                FROM preflop_rfi
                WHERE game_type = ? AND position = ?
                GROUP BY action
            """, (game_type, position))
            rows = c.fetchall()
            total_raise = sum(cnt for a, cnt in rows if a == "R")
            total_limp  = sum(cnt for a, cnt in rows if a == "L")
            total_hands = 169
            return {
                "raise_pct": round(total_raise / total_hands * 100, 1),
                "limp_pct":  round(total_limp  / total_hands * 100, 1),
                "position": position,
                "game_type": game_type,
            }
        finally:
            conn.close()

    # ─── POSTFLOP ────────────────────────────────────────────────────────────

    def get_postflop_action(self, hand_category, board_texture,
                             position_type, stack_to_pot="normal"):
        """
        Retorna ação postflop baseada em categoria de mão e textura do board.

        hand_category: str — ex: "top_pair_top_kicker", "flush_draw"
        board_texture: str — ex: "dry", "wet", "paired_board", "monotone"
        position_type: str — "IP" (in position) ou "OOP" (out of position)
        stack_to_pot: str — "high" (>10x), "normal" (3-10x), "low" (<3x)
        """
        conn = self._connect()
        c = conn.cursor()

        try:
            # Busca exata
            c.execute("""
                SELECT action, reason FROM postflop_heuristics
                WHERE hand_category = ?
                  AND (board_texture = ? OR board_texture = 'any')
                  AND (position_type = ? OR position_type = 'any')
                  AND (stack_to_pot = ? OR stack_to_pot = 'any')
                ORDER BY
                    CASE WHEN board_texture = ? THEN 0 ELSE 1 END,
                    CASE WHEN position_type = ? THEN 0 ELSE 1 END
                LIMIT 1
            """, (hand_category, board_texture, position_type, stack_to_pot,
                  board_texture, position_type))
            row = c.fetchone()

            if not row:
                # Fallback genérico
                c.execute("""
                    SELECT action, reason FROM postflop_heuristics
                    WHERE hand_category = ?
                    LIMIT 1
                """, (hand_category,))
                row = c.fetchone()

        finally:
            conn.close()

        if row:
            action, reason = row
            label, desc = ACTION_LABELS.get(action, (action, ""))
            return {
                "action": action,
                "label": label,
                "description": desc,
                "reason": reason,
                "hand_category": hand_category,
            }

        return {
            "action": "CHECK_CALL",
            "label": ACTION_LABELS["CHECK_CALL"][0],
            "description": ACTION_LABELS["CHECK_CALL"][1],
            "reason": "Sem dados específicos — jogue cautelosamente",
            "hand_category": hand_category,
        }

    # ─── POT ODDS ────────────────────────────────────────────────────────────

    def get_pot_odds(self, bet_to_call, pot_size):
        """
        Calcula pot odds e retorna se deve chamar.

        bet_to_call: float — valor da aposta a pagar
        pot_size: float — tamanho atual do pote (antes da aposta)
        """
        if pot_size <= 0 or bet_to_call <= 0:
            return {"error": "Valores inválidos"}

        required_equity = bet_to_call / (pot_size + bet_to_call * 2)
        pot_odds_pct = (bet_to_call / (pot_size + bet_to_call)) * 100

        # Busca decisão no banco
        conn = self._connect()
        c = conn.cursor()
        try:
            bet_pct = (bet_to_call / pot_size) * 100
            c.execute("""
                SELECT decision FROM pot_odds
                ORDER BY ABS(bet_to_call - ?) ASC
                LIMIT 1
            """, (bet_to_call,))
            row = c.fetchone()
            db_decision = row[0] if row else None
        finally:
            conn.close()

        # Decisão baseada em equity
        if required_equity < 0.25:
            decision = "✅ CHAME — Odds excelentes"
        elif required_equity < 0.33:
            decision = "✅ CHAME — Boas odds"
        elif required_equity < 0.40:
            decision = "🟡 DEPENDE — Analise sua equity"
        elif required_equity < 0.45:
            decision = "🔴 BORDERLINE — Precisa de draw forte"
        else:
            decision = "🔴 FOLD — Odds ruins sem equity"

        return {
            "bet_to_call": bet_to_call,
            "pot_size": pot_size,
            "required_equity_pct": round(required_equity * 100, 1),
            "pot_odds_pct": round(pot_odds_pct, 1),
            "decision": decision,
        }

    # ─── SPR (Stack to Pot Ratio) ─────────────────────────────────────────────

    def get_spr_advice(self, effective_stack, pot_size):
        """Conselho baseado no SPR."""
        if pot_size <= 0:
            return None

        spr = effective_stack / pot_size

        if spr < 1:
            return {
                "spr": round(spr, 2),
                "category": "SHALLOW",
                "advice": "🔴 SPR muito baixo — Vá all-in com qualquer par ou melhor",
                "commitment": "Committed com qualquer par ou TP"
            }
        elif spr < 3:
            return {
                "spr": round(spr, 2),
                "category": "LOW",
                "advice": "🟡 SPR baixo — Commited com TP ou melhor, bluffs não valem",
                "commitment": "Top pair+ é suficiente para ir all-in"
            }
        elif spr < 6:
            return {
                "spr": round(spr, 2),
                "category": "MEDIUM",
                "advice": "🟢 SPR médio — Jogue por valor com mãos fortes, controle com médias",
                "commitment": "Duas duplas ou melhor para grande comprometimento"
            }
        elif spr < 13:
            return {
                "spr": round(spr, 2),
                "category": "HIGH",
                "advice": "🔵 SPR alto — Precise de mãos muito fortes para ir all-in",
                "commitment": "Set ou melhor para comprometimento máximo"
            }
        else:
            return {
                "spr": round(spr, 2),
                "category": "VERY_HIGH",
                "advice": "🔵 SPR muito alto — Deep stacks, jogue com sets+, bluffs têm equity",
                "commitment": "Straight/Flush+ para all-in"
            }

    # ─── POSIÇÕES ─────────────────────────────────────────────────────────────

    def get_positions(self, game_type):
        """Retorna lista de posições para o tipo de mesa."""
        conn = self._connect()
        c = conn.cursor()
        try:
            c.execute("""
                SELECT seat_number, position_name, position_type
                FROM table_positions
                WHERE game_type = ?
                ORDER BY seat_number
            """, (game_type,))
            return [
                {"seat": r[0], "name": r[1], "type": r[2]}
                for r in c.fetchall()
            ]
        finally:
            conn.close()

    # ─── ANÁLISE COMPLETA ─────────────────────────────────────────────────────

    def full_analysis(self, hand, position, game_type, situation,
                      raiser_pos, hand_category, board_texture,
                      position_type, stack, pot, board_cards):
        """
        Análise completa pré e pós-flop com todas as métricas.
        """
        result = {
            "hand": hand,
            "position": position,
            "game_type": game_type,
            "street": STREET_LABELS.get(len(board_cards), "River"),
        }

        # Preflop
        if len(board_cards) == 0:
            preflop = self.get_preflop_action(hand, position, game_type,
                                               situation, raiser_pos)
            result["preflop"] = preflop
            result["primary_action"] = preflop

        # Postflop
        else:
            postflop = self.get_postflop_action(hand_category, board_texture,
                                                 position_type)
            result["postflop"] = postflop
            result["primary_action"] = postflop

        # SPR
        if stack and pot and pot > 0:
            result["spr"] = self.get_spr_advice(float(stack), float(pot))

        return result
