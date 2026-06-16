"""
app.py
GTO Poker Advisor — Streamlit mobile-first
Câmera do celular lê a mesa → banco de dados retorna decisão GTO
"""

import streamlit as st
import os
import sys

# ─── CONFIG DA PÁGINA ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GTO Poker Advisor",
    page_icon="🃏",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS MOBILE-FIRST ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Reset e variáveis */
  :root {
    --green:  #00c853;
    --red:    #f44336;
    --blue:   #2196f3;
    --yellow: #ffc107;
    --bg:     #0d1117;
    --card:   #161b22;
    --border: #30363d;
    --text:   #e6edf3;
    --muted:  #8b949e;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Remove padding padrão */
  .block-container { padding: 0.5rem 0.75rem 2rem !important; max-width: 100% !important; }
  header { display: none !important; }

  /* Cards */
  .gto-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }

  /* Resultado principal */
  .action-green  { background: #0a2e1a; border: 2px solid var(--green); border-radius: 12px; padding: 16px; text-align: center; margin: 10px 0; }
  .action-red    { background: #2e0a0a; border: 2px solid var(--red);   border-radius: 12px; padding: 16px; text-align: center; margin: 10px 0; }
  .action-blue   { background: #0a1e2e; border: 2px solid var(--blue);  border-radius: 12px; padding: 16px; text-align: center; margin: 10px 0; }
  .action-yellow { background: #2e220a; border: 2px solid var(--yellow);border-radius: 12px; padding: 16px; text-align: center; margin: 10px 0; }
  .action-fire   { background: #2e0a2e; border: 2px solid #e040fb;      border-radius: 12px; padding: 16px; text-align: center; margin: 10px 0; }

  .action-label { font-size: 1.6rem; font-weight: 800; letter-spacing: 1px; }
  .action-desc  { font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
  .action-reason { font-size: 0.85rem; color: var(--text); margin-top: 8px; }

  /* Cartas */
  .card-display { display: inline-block; background: white; color: black;
    border-radius: 6px; padding: 4px 8px; margin: 3px;
    font-size: 1.1rem; font-weight: 700; min-width: 36px; text-align: center; }
  .card-red { color: #c62828 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--card); border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 12px; font-size: 0.85rem; }

  /* Inputs */
  .stTextInput input, .stSelectbox select, .stNumberInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
  }

  /* Botão principal */
  .stButton > button {
    width: 100%;
    background: var(--green) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 12px !important;
    font-size: 1rem !important;
  }

  /* Labels */
  label { color: var(--muted) !important; font-size: 0.8rem !important; }

  /* Stat boxes */
  .stat-row { display: flex; gap: 8px; margin: 8px 0; }
  .stat-box { flex: 1; background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px; text-align: center; }
  .stat-val { font-size: 1.2rem; font-weight: 700; color: var(--green); }
  .stat-lbl { font-size: 0.7rem; color: var(--muted); }

  /* Street badge */
  .street-badge { display: inline-block; background: #21262d;
    border-radius: 20px; padding: 3px 12px; font-size: 0.75rem;
    color: var(--muted); margin-bottom: 8px; }

  /* Divider */
  hr { border-color: var(--border) !important; margin: 10px 0 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── INICIALIZA DB SE NECESSÁRIO ──────────────────────────────────────────────
@st.cache_resource
def init_database():
    if not os.path.exists("gto_poker.db"):
        try:
            import build_gto_db
            build_gto_db.create_database()
            return True
        except Exception as e:
            return False
    return True

db_ok = init_database()

# ─── IMPORTS DOS MÓDULOS ──────────────────────────────────────────────────────
try:
    from gto_engine import GTOEngine
    from hand_evaluator import HandEvaluator, RANKS
    from card_detector import CardDetector, parse_cards_string
    engine   = GTOEngine()
    evaluator = HandEvaluator()
    detector  = CardDetector()
    modules_ok = True
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    modules_ok = False
except Exception as e:
    st.error(f"Erro ao inicializar módulos: {e}")
    modules_ok = False

# ─── HELPERS ─────────────────────────────────────────────────────────────────
SUIT_COLORS_HTML = {"h": "red", "d": "red", "s": "black", "c": "green"}
SUIT_SYMBOLS     = {"h": "♥", "d": "♦", "s": "♠", "c": "♣"}

def render_card(card_str):
    if not card_str or len(card_str) < 2:
        return ""
    rank = card_str[:-1]
    suit = card_str[-1].lower()
    symbol = SUIT_SYMBOLS.get(suit, suit)
    color_class = "card-red" if suit in ("h", "d") else ""
    return f'<span class="card-display {color_class}">{rank}{symbol}</span>'

def render_cards(cards_list):
    return "".join(render_card(c) for c in cards_list)

def action_class(action_code):
    if action_code in ("R", "3B", "BET_BIG", "CHECK_RAISE"):
        return "action-fire" if "3B" in action_code or "RAISE" in action_code else "action-green"
    if action_code in ("C", "CHECK_CALL", "L"):
        return "action-blue"
    if action_code in ("F", "CHECK_FOLD"):
        return "action-red"
    if action_code in ("BET_MED", "BET_SMALL"):
        return "action-yellow"
    return "action-blue"

def positions_for(game_type):
    if game_type == "6max":
        return ["BTN", "CO", "HJ", "MP", "SB", "BB"]
    return ["BTN", "CO", "HJ", "MP2", "MP1", "UTG1", "UTG", "SB", "BB"]

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 10px 0 4px;">
  <span style="font-size:2rem;">🃏</span>
  <div style="font-size:1.2rem; font-weight:800; letter-spacing:1px;">GTO POKER</div>
  <div style="font-size:0.7rem; color:#8b949e;">Advisor • 6max & 9max</div>
</div>
""", unsafe_allow_html=True)

if not modules_ok:
    st.stop()

# ─── ABAS PRINCIPAIS ─────────────────────────────────────────────────────────
tab_cam, tab_manual, tab_odds = st.tabs(["📷 Câmera", "✍️ Manual", "📊 Odds"])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1: CÂMERA
# ══════════════════════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown('<div class="gto-card">', unsafe_allow_html=True)
    st.markdown("**📷 Aponte para a mesa de poker**")
    st.markdown('<div style="font-size:0.8rem;color:#8b949e;">Capture a tela do app com suas cartas visíveis</div>', unsafe_allow_html=True)

    cam_image = st.camera_input("", label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

    if cam_image:
        img_bytes = cam_image.getvalue()

        with st.spinner("🔍 Analisando mesa..."):
            detected = detector.detect_cards_from_image(img_bytes)

        hole_cards  = detected.get("hole_cards", [])
        board_cards = detected.get("board_cards", [])

        st.markdown('<div class="gto-card">', unsafe_allow_html=True)
        st.markdown("**Cartas detectadas**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Hole cards")
            if hole_cards:
                st.markdown(render_cards(hole_cards), unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#8b949e;font-size:0.8rem;">Nenhuma detectada</span>', unsafe_allow_html=True)

        with col2:
            st.markdown("Board")
            if board_cards:
                st.markdown(render_cards(board_cards), unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#8b949e;font-size:0.8rem;">Preflop</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Configurações rápidas
        st.markdown('<div class="gto-card">', unsafe_allow_html=True)
        col_g, col_p = st.columns(2)
        with col_g:
            game_type_cam = st.selectbox("Mesa", ["6max", "9max"], key="cam_game")
        with col_p:
            positions_cam = positions_for(game_type_cam)
            position_cam  = st.selectbox("Minha posição", positions_cam, key="cam_pos")

        col_s, col_pot = st.columns(2)
        with col_s:
            stack_cam = st.number_input("Stack (BB)", min_value=1, value=100, key="cam_stack")
        with col_pot:
            pot_cam   = st.number_input("Pot (BB)",   min_value=0, value=0,   key="cam_pot")

        situation_cam = st.selectbox(
            "Situação", ["rfi", "bb_defense", "3bet"],
            format_func=lambda x: {"rfi": "Abrir pot (RFI)", "bb_defense": "BB defendendo", "3bet": "3-bet"}[x],
            key="cam_sit"
        )

        raiser_cam = None
        if situation_cam != "rfi":
            others = [p for p in positions_cam if p != position_cam]
            raiser_cam = st.selectbox("Posição do raiser", others, key="cam_raiser")

        st.markdown('</div>', unsafe_allow_html=True)

        if hole_cards and len(hole_cards) >= 2:
            hand_norm = detector.normalize_hand(hole_cards[0], hole_cards[1])

            if hand_norm:
                # ── Análise GTO ───────────────────────────────────────────
                board_tex = evaluator.classify_board_texture(board_cards)
                hand_eval = evaluator.evaluate(hole_cards, board_cards)
                hand_cat  = hand_eval.get("category", "unknown")

                ip_positions = ["BTN", "CO", "HJ"]
                pos_type = "IP" if position_cam in ip_positions else "OOP"

                if board_cards:
                    gto = engine.get_postflop_action(hand_cat, board_tex, pos_type)
                    street = {3: "Flop", 4: "Turn", 5: "River"}.get(len(board_cards), "Postflop")
                else:
                    gto = engine.get_preflop_action(hand_norm, position_cam, game_type_cam,
                                                     situation_cam, raiser_cam)
                    street = "Preflop"

                spr_data = engine.get_spr_advice(stack_cam, pot_cam) if pot_cam > 0 else None

                # ── Exibe resultado ───────────────────────────────────────
                st.markdown(f'<div class="street-badge">{street}</div>', unsafe_allow_html=True)

                action_code = gto.get("action", "?")
                css_class   = action_class(action_code)

                st.markdown(f"""
                <div class="{css_class}">
                  <div class="action-label">{gto.get("label", action_code)}</div>
                  <div class="action-desc">{gto.get("description", "")}</div>
                  <div class="action-reason">{gto.get("reason", gto.get("source",""))}</div>
                </div>
                """, unsafe_allow_html=True)

                # Métricas
                outs_data = evaluator.get_outs(hole_cards, board_cards) if board_cards else None

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""<div class="stat-box">
                      <div class="stat-val">{hand_norm}</div>
                      <div class="stat-lbl">Mão</div></div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="stat-box">
                      <div class="stat-val">{board_tex.upper()}</div>
                      <div class="stat-lbl">Board</div></div>""", unsafe_allow_html=True)
                with col3:
                    eq = outs_data["equity_pct"] if outs_data else "—"
                    eq_str = f"{eq}%" if isinstance(eq, (int, float)) else eq
                    st.markdown(f"""<div class="stat-box">
                      <div class="stat-val">{eq_str}</div>
                      <div class="stat-lbl">Equity</div></div>""", unsafe_allow_html=True)

                if spr_data:
                    st.markdown(f'<div style="margin-top:8px;">{spr_data["advice"]}</div>', unsafe_allow_html=True)
            else:
                st.warning("Não foi possível normalizar a mão. Corrija as cartas abaixo.")
                # Permite correção manual
                c1 = st.text_input("Carta 1 (ex: Ah)", value=hole_cards[0] if hole_cards else "", key="cam_fix1")
                c2 = st.text_input("Carta 2 (ex: Ks)", value=hole_cards[1] if len(hole_cards)>1 else "", key="cam_fix2")
        else:
            st.info("💡 Segure o celular firme e aponte para suas 2 cartas. Se a câmera não detectar, use a aba **✍️ Manual**.")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2: ENTRADA MANUAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:

    # ── Configurações ────────────────────────────────────────────────────────
    st.markdown('<div class="gto-card">', unsafe_allow_html=True)
    col_g2, col_p2 = st.columns(2)
    with col_g2:
        game_type = st.selectbox("Tipo de mesa", ["6max", "9max"], key="m_game")
    with col_p2:
        positions_list = positions_for(game_type)
        my_position    = st.selectbox("Minha posição", positions_list, key="m_pos")

    col_s2, col_pot2 = st.columns(2)
    with col_s2:
        my_stack = st.number_input("Meu stack (BB)", min_value=1, value=100, key="m_stack")
    with col_pot2:
        pot_size = st.number_input("Pot atual (BB)", min_value=0, value=0, key="m_pot")

    situation = st.selectbox(
        "Situação preflop",
        ["rfi", "bb_defense", "3bet"],
        format_func=lambda x: {
            "rfi":        "🟢 Abrir pote (RFI — primeiro a entrar)",
            "bb_defense": "🔵 BB defendendo vs raise",
            "3bet":       "🔥 Considero 3-bet"
        }[x],
        key="m_sit"
    )

    raiser_pos = None
    if situation != "rfi":
        others = [p for p in positions_list if p != my_position]
        raiser_pos = st.selectbox("Quem abriu?", others, key="m_raiser")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Cartas ───────────────────────────────────────────────────────────────
    st.markdown('<div class="gto-card">', unsafe_allow_html=True)
    st.markdown("**Suas cartas (hole cards)**")
    st.markdown('<div style="font-size:0.75rem;color:#8b949e;">Ex: Ah Ks &nbsp;|&nbsp; Td 9d &nbsp;|&nbsp; 2c 2h</div>', unsafe_allow_html=True)

    hole_input = st.text_input("", placeholder="Ah Ks", key="m_hole", label_visibility="collapsed")

    st.markdown("**Cartas do board** (opcional)")
    st.markdown('<div style="font-size:0.75rem;color:#8b949e;">Flop: 3 cartas &nbsp;|&nbsp; Turn: 4 &nbsp;|&nbsp; River: 5</div>', unsafe_allow_html=True)
    board_input = st.text_input("", placeholder="2h 7d Tc", key="m_board", label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Botão analisar ────────────────────────────────────────────────────────
    analyze = st.button("🔍 ANALISAR", key="m_analyze")

    if analyze or (hole_input and len(hole_input) >= 4):
        hole_cards_m  = parse_cards_string(hole_input)
        board_cards_m = parse_cards_string(board_input)

        if len(hole_cards_m) < 2:
            st.error("⚠️ Digite pelo menos 2 cartas. Ex: **Ah Ks**")
            st.stop()

        hand_norm_m = detector.normalize_hand(hole_cards_m[0], hole_cards_m[1])

        if not hand_norm_m:
            st.error("⚠️ Cartas inválidas. Use formato: rank + naipe (Ah, Ks, Td, 9c)")
            st.stop()

        # ── Preview das cartas ────────────────────────────────────────────
        st.markdown('<div class="gto-card">', unsafe_allow_html=True)
        col_hc, col_bc = st.columns(2)
        with col_hc:
            st.markdown("Hole cards")
            st.markdown(render_cards(hole_cards_m), unsafe_allow_html=True)
        with col_bc:
            if board_cards_m:
                st.markdown("Board")
                st.markdown(render_cards(board_cards_m), unsafe_allow_html=True)
            else:
                st.markdown("Board")
                st.markdown('<span style="color:#8b949e;font-size:0.8rem;">Preflop</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Análise ───────────────────────────────────────────────────────
        board_tex_m = evaluator.classify_board_texture(board_cards_m)
        hand_eval_m = evaluator.evaluate(hole_cards_m, board_cards_m)
        hand_cat_m  = hand_eval_m.get("category", "unknown")

        ip_pos = ["BTN", "CO", "HJ"]
        pos_type_m = "IP" if my_position in ip_pos else "OOP"

        if board_cards_m:
            gto_m  = engine.get_postflop_action(hand_cat_m, board_tex_m, pos_type_m)
            street_m = {3: "🃏 Flop", 4: "🎴 Turn", 5: "🏁 River"}.get(len(board_cards_m), "Postflop")
        else:
            gto_m  = engine.get_preflop_action(hand_norm_m, my_position, game_type,
                                                situation, raiser_pos)
            street_m = "🎯 Preflop"

        spr_m    = engine.get_spr_advice(my_stack, pot_size) if pot_size > 0 else None
        outs_m   = evaluator.get_outs(hole_cards_m, board_cards_m) if board_cards_m else None

        # ── Resultado principal ───────────────────────────────────────────
        st.markdown(f'<div class="street-badge">{street_m}</div>', unsafe_allow_html=True)

        action_m_code = gto_m.get("action", "?")
        css_m = action_class(action_m_code)

        st.markdown(f"""
        <div class="{css_m}">
          <div class="action-label">{gto_m.get("label", action_m_code)}</div>
          <div class="action-desc">{gto_m.get("description","")}</div>
          <div class="action-reason">{gto_m.get("reason", gto_m.get("source",""))}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Estatísticas ──────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="stat-box">
              <div class="stat-val">{hand_norm_m}</div>
              <div class="stat-lbl">Mão GTO</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-box">
              <div class="stat-val">{pos_type_m}</div>
              <div class="stat-lbl">Posição</div></div>""", unsafe_allow_html=True)
        with col3:
            desc_m = hand_eval_m.get("description","—")[:10]
            st.markdown(f"""<div class="stat-box">
              <div class="stat-val" style="font-size:0.85rem;">{desc_m}</div>
              <div class="stat-lbl">Mão</div></div>""", unsafe_allow_html=True)

        if board_cards_m:
            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown(f"""<div class="stat-box">
                  <div class="stat-val">{board_tex_m}</div>
                  <div class="stat-lbl">Textura</div></div>""", unsafe_allow_html=True)
            with col5:
                outs_n = outs_m["outs"] if outs_m else 0
                st.markdown(f"""<div class="stat-box">
                  <div class="stat-val">{outs_n}</div>
                  <div class="stat-lbl">Outs</div></div>""", unsafe_allow_html=True)
            with col6:
                eq_m = outs_m["equity_pct"] if outs_m else 0
                st.markdown(f"""<div class="stat-box">
                  <div class="stat-val">{eq_m}%</div>
                  <div class="stat-lbl">Equity</div></div>""", unsafe_allow_html=True)

        # ── SPR ───────────────────────────────────────────────────────────
        if spr_m:
            st.markdown(f"""
            <div class="gto-card" style="margin-top:10px;">
              <div style="font-size:0.75rem;color:#8b949e;">SPR — Stack to Pot Ratio</div>
              <div style="font-size:1.2rem;font-weight:700;">{spr_m['spr']}x</div>
              <div style="font-size:0.85rem;margin-top:4px;">{spr_m['advice']}</div>
              <div style="font-size:0.75rem;color:#8b949e;margin-top:4px;">{spr_m['commitment']}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Informações de range ──────────────────────────────────────────
        if not board_cards_m:
            try:
                range_stats = engine.get_preflop_range_stats(my_position, game_type)
                st.markdown(f"""
                <div class="gto-card">
                  <div style="font-size:0.75rem;color:#8b949e;">Range — {my_position} ({game_type})</div>
                  <div style="margin-top:6px;">
                    <span style="color:#00c853;font-weight:700;">{range_stats['raise_pct']}%</span>
                    <span style="color:#8b949e;font-size:0.8rem;"> raise &nbsp;|&nbsp; </span>
                    <span style="color:#ffc107;font-weight:700;">{range_stats['limp_pct']}%</span>
                    <span style="color:#8b949e;font-size:0.8rem;"> limp</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            except:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# ABA 3: POT ODDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_odds:
    st.markdown('<div class="gto-card">', unsafe_allow_html=True)
    st.markdown("**Calculadora de Pot Odds**")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        bet_call = st.number_input("Aposta a pagar (BB)", min_value=0.5, value=10.0,
                                    step=0.5, key="o_bet")
    with col_o2:
        pot_o    = st.number_input("Pote atual (BB)", min_value=1.0, value=30.0,
                                    step=1.0, key="o_pot")

    outs_input = st.number_input("Meus outs (0 se não souber)", min_value=0, max_value=20,
                                  value=0, key="o_outs")

    calc_odds = st.button("📊 CALCULAR ODDS", key="o_calc")
    st.markdown('</div>', unsafe_allow_html=True)

    if calc_odds or True:
        if bet_call > 0 and pot_o > 0:
            odds = engine.get_pot_odds(bet_call, pot_o)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div class="stat-box">
                  <div class="stat-val">{odds['required_equity_pct']}%</div>
                  <div class="stat-lbl">Equity mínima</div></div>""", unsafe_allow_html=True)
            with col2:
                ratio = round(pot_o / bet_call, 1)
                st.markdown(f"""<div class="stat-box">
                  <div class="stat-val">{ratio}:1</div>
                  <div class="stat-lbl">Pot odds</div></div>""", unsafe_allow_html=True)

            css_odds = "action-green" if "CALL" in odds["decision"] else (
                "action-yellow" if "DEPENDE" in odds["decision"] else "action-red")
            st.markdown(f"""
            <div class="{css_odds}">
              <div class="action-label" style="font-size:1.2rem;">{odds['decision']}</div>
            </div>
            """, unsafe_allow_html=True)

            if outs_input > 0:
                st.markdown('<div class="gto-card">', unsafe_allow_html=True)
                st.markdown("**Equity pelo número de outs (Regra 4/2)**")

                eq_flop  = min(outs_input * 4, 100)
                eq_river = min(outs_input * 2, 100)

                col3, col4 = st.columns(2)
                with col3:
                    clr = "#00c853" if eq_flop >= odds["required_equity_pct"] else "#f44336"
                    st.markdown(f"""<div class="stat-box">
                      <div class="stat-val" style="color:{clr};">{eq_flop}%</div>
                      <div class="stat-lbl">Equity no Flop</div></div>""", unsafe_allow_html=True)
                with col4:
                    clr = "#00c853" if eq_river >= odds["required_equity_pct"] else "#f44336"
                    st.markdown(f"""<div class="stat-box">
                      <div class="stat-val" style="color:{clr};">{eq_river}%</div>
                      <div class="stat-lbl">Equity no Turn</div></div>""", unsafe_allow_html=True)

                if eq_flop >= odds["required_equity_pct"]:
                    st.markdown(f'<div style="color:#00c853;margin-top:8px;">✅ Com {outs_input} outs você tem equity suficiente para chamar!</div>', unsafe_allow_html=True)
                else:
                    needed = round(odds["required_equity_pct"] / 4)
                    st.markdown(f'<div style="color:#f44336;margin-top:8px;">❌ Precisa de ≥ {needed} outs para justificar o call</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Referência rápida
            with st.expander("📖 Referência de Outs"):
                st.markdown("""
| Outs | Draw | Equity Flop |
|------|------|-------------|
| 15 | Flush + OESD | ~60% |
| 9  | Flush draw | ~36% |
| 8  | OESD | ~32% |
| 6  | Dois overcards | ~24% |
| 4  | Gutshot | ~16% |
| 2  | Pocket pair → set | ~8% |
                """)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px 0 10px;color:#30363d;font-size:0.7rem;">
  GTO Poker Advisor • Uso educacional apenas<br>
  Baseado em solver ranges públicos • Não garante resultado
</div>
""", unsafe_allow_html=True)
