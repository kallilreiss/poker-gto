"""
app.py — GTO Poker Advisor (PokerStars Edition)
Mobile-first Streamlit app com detecção de câmera melhorada.
"""

import streamlit as st
import os
import numpy as np

st.set_page_config(
    page_title="GTO Poker",
    page_icon="🃏",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --green:#00c853; --red:#f44336; --blue:#2196f3;
  --yellow:#ffc107; --bg:#0d1117; --card:#161b22;
  --border:#30363d; --text:#e6edf3; --muted:#8b949e;
}
html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif!important;
  background:var(--bg)!important;color:var(--text)!important;}
.block-container{padding:0.4rem 0.6rem 3rem!important;max-width:100%!important;}
header,footer{display:none!important;}

/* Cards */
.box{background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px;margin-bottom:10px;}

/* Resultado */
.res{border-radius:12px;padding:18px;text-align:center;margin:8px 0;}
.res-green{background:#0a2e1a;border:2px solid var(--green);}
.res-red  {background:#2e0a0a;border:2px solid var(--red);}
.res-blue {background:#0a1e2e;border:2px solid var(--blue);}
.res-yel  {background:#2e220a;border:2px solid var(--yellow);}
.res-fire {background:#200a2e;border:2px solid #e040fb;}
.res-lbl{font-size:1.55rem;font-weight:800;letter-spacing:1px;}
.res-sub{font-size:0.85rem;color:var(--muted);margin-top:3px;}
.res-why{font-size:0.82rem;margin-top:7px;}

/* Cartas */
.cd{display:inline-block;background:#fff;color:#111;border-radius:6px;
  padding:5px 9px;margin:3px;font-size:1.05rem;font-weight:700;
  min-width:38px;text-align:center;border:1px solid #ccc;}
.cd-r{color:#c00!important;}

/* Stats */
.sbox{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:10px;text-align:center;}
.sv{font-size:1.15rem;font-weight:700;color:var(--green);}
.sl{font-size:0.68rem;color:var(--muted);}

/* Badge */
.badge{display:inline-block;background:#21262d;border-radius:20px;
  padding:3px 12px;font-size:0.72rem;color:var(--muted);margin-bottom:6px;}

/* Picker de cartas */
.rank-btn button{background:#21262d!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:6px!important;font-weight:700!important;
  padding:6px!important;min-width:38px!important;}
.suit-h button,.suit-d button{color:#c00!important;font-weight:800!important;}
.suit-s button,.suit-c button{color:#fff!important;font-weight:800!important;}
.sel-card{background:#00c853!important;color:#000!important;border-color:#00c853!important;}

/* Botão principal */
.stButton>button{width:100%;background:var(--green)!important;color:#000!important;
  font-weight:700!important;border-radius:10px!important;border:none!important;
  padding:12px!important;font-size:1rem!important;}

label{color:var(--muted)!important;font-size:0.78rem!important;}
hr{border-color:var(--border)!important;margin:8px 0!important;}

/* Camera */
.stCameraInput>div{border-radius:12px!important;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

# ─── INIT DB ─────────────────────────────────────────────────────────────────
@st.cache_resource
def init_db():
    if not os.path.exists("gto_poker.db"):
        import build_gto_db
        build_gto_db.create_database()
    return True

init_db()

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
try:
    from gto_engine    import GTOEngine
    from hand_evaluator import HandEvaluator, RANKS
    from card_detector  import CardDetector, parse_cards_string
    engine    = GTOEngine()
    evaluator = HandEvaluator()
    detector  = CardDetector()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

# ─── HELPERS ─────────────────────────────────────────────────────────────────
SUIT_SYM = {"h":"♥","d":"♦","s":"♠","c":"♣"}
SUIT_RED  = {"h","d"}

def card_html(c):
    if not c or len(c) < 2: return ""
    r, s = c[:-1], c[-1].lower()
    sym = SUIT_SYM.get(s, s)
    cls = "cd cd-r" if s in SUIT_RED else "cd"
    return f'<span class="{cls}">{r}{sym}</span>'

def cards_html(lst):
    return "".join(card_html(c) for c in lst)

def res_class(a):
    if a in ("3B","CHECK_RAISE"):      return "res res-fire"
    if a in ("R","BET_BIG"):           return "res res-green"
    if a in ("BET_MED","BET_SMALL","L"): return "res res-yel"
    if a in ("C","CHECK_CALL"):        return "res res-blue"
    return "res res-red"

def positions(gt):
    return (["BTN","CO","HJ","MP","SB","BB"] if gt=="6max"
            else ["BTN","CO","HJ","MP2","MP1","UTG1","UTG","SB","BB"])

def show_result(gto, street="Preflop", spr=None, outs=None):
    st.markdown(f'<div class="badge">{street}</div>', unsafe_allow_html=True)
    a = gto.get("action","?")
    st.markdown(f"""
    <div class="{res_class(a)}">
      <div class="res-lbl">{gto.get("label", a)}</div>
      <div class="res-sub">{gto.get("description","")}</div>
      <div class="res-why">{gto.get("reason", gto.get("source",""))}</div>
    </div>""", unsafe_allow_html=True)
    if spr:
        st.markdown(f'<div class="box" style="margin-top:6px;">'
                    f'<span style="font-size:0.7rem;color:var(--muted);">SPR</span> '
                    f'<b>{spr["spr"]}x</b> — {spr["advice"]}</div>',
                    unsafe_allow_html=True)
    if outs and outs["outs"] > 0:
        st.markdown(f'<div class="box"><b>{outs["outs"]} outs</b> → '
                    f'~{outs["equity_pct"]}% equity ({outs["draw_type"]})</div>',
                    unsafe_allow_html=True)

def run_analysis(hole, board, pos, gt, situation, raiser, stack, pot):
    if len(hole) < 2:
        st.warning("Precisa de 2 hole cards."); return

    hand_norm = detector.normalize_hand(hole[0], hole[1])
    if not hand_norm:
        st.error("Cartas inválidas."); return

    board_tex = evaluator.classify_board_texture(board)
    hand_eval = evaluator.evaluate(hole, board)
    hand_cat  = hand_eval.get("category","unknown")
    ip_pos    = {"BTN","CO","HJ"}
    pos_type  = "IP" if pos in ip_pos else "OOP"
    spr_d     = engine.get_spr_advice(stack, pot) if pot > 0 else None
    outs_d    = evaluator.get_outs(hole, board) if board else None

    st.markdown('<div class="box">', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Hole cards**")
        st.markdown(cards_html(hole), unsafe_allow_html=True)
    with c2:
        if board:
            st.markdown("**Board**")
            st.markdown(cards_html(board), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if board:
        gto  = engine.get_postflop_action(hand_cat, board_tex, pos_type)
        street = {3:"🃏 Flop",4:"🎴 Turn",5:"🏁 River"}.get(len(board),"Postflop")
    else:
        gto  = engine.get_preflop_action(hand_norm, pos, gt, situation, raiser)
        street = "🎯 Preflop"

    show_result(gto, street, spr_d, outs_d)

    # Métricas
    cols = st.columns(3)
    stats = [
        (hand_norm, "Mão GTO"),
        (hand_eval.get("description","")[:12], "Força"),
        (board_tex if board else pos_type, "Board/Pos"),
    ]
    for col,(val,lbl) in zip(cols,stats):
        col.markdown(f'<div class="sbox"><div class="sv">{val}</div>'
                     f'<div class="sl">{lbl}</div></div>', unsafe_allow_html=True)

# ─── ESTADO ──────────────────────────────────────────────────────────────────
if "hole"  not in st.session_state: st.session_state.hole  = []
if "board" not in st.session_state: st.session_state.board = []
if "cam_analyzed" not in st.session_state: st.session_state.cam_analyzed = False

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:10px 0 6px;">
  <span style="font-size:2rem;">🃏</span>
  <div style="font-size:1.2rem;font-weight:800;">GTO POKER ADVISOR</div>
  <div style="font-size:0.68rem;color:#8b949e;">PokerStars • 6max & 9max</div>
</div>""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_cam, tab_pick, tab_odds = st.tabs(["📷 Câmera", "🃏 Selecionar", "📊 Odds"])

# ══════════════════════════════════════════════════════════════════════════════
# ABA CÂMERA
# ══════════════════════════════════════════════════════════════════════════════
with tab_cam:

    # Configurações inline
    st.markdown('<div class="box">', unsafe_allow_html=True)
    cg1, cg2 = st.columns(2)
    with cg1: gt_c = st.selectbox("Mesa", ["6max","9max"], key="cgt")
    with cg2:
        pos_list_c = positions(gt_c)
        pos_c = st.selectbox("Minha posição", pos_list_c, key="cpos")

    cs1, cs2 = st.columns(2)
    with cs1: stk_c = st.number_input("Stack (BB)", 1, 5000, 100, key="cstk")
    with cs2: pot_c = st.number_input("Pot (BB)",   0, 5000,   0, key="cpot")

    sit_c = st.selectbox("Situação", ["rfi","bb_defense","3bet"],
        format_func=lambda x:{"rfi":"Abrir pot","bb_defense":"BB vs raise","3bet":"3-bet"}[x],
        key="csit")
    rai_c = None
    if sit_c != "rfi":
        others_c = [p for p in pos_list_c if p != pos_c]
        rai_c = st.selectbox("Raiser", others_c, key="crai")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── DICAS DE CAPTURA ────────────────────────────────────────────────────
    with st.expander("💡 Como capturar melhor"):
        st.markdown("""
**Para melhor detecção no PokerStars:**
1. **Tela brilhante** — brilho do celular no máximo
2. **Sem reflexo** — segure perpendicular à tela
3. **Cartas visíveis** — as 2 suas + board completo apareça
4. **Boa iluminação** — evite ambientes escuros
5. **Câmera parada** — espere focar antes de capturar

Se a detecção falhar, use a aba **🃏 Selecionar** para escolher as cartas manualmente.
        """)

    # ── CÂMERA ───────────────────────────────────────────────────────────────
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("**📷 Aponte para a tela do PokerStars**")
    cam_img = st.camera_input("", label_visibility="collapsed", key="cam")
    st.markdown('</div>', unsafe_allow_html=True)

    if cam_img:
        img_bytes = cam_img.getvalue()

        with st.spinner("🔍 Processando imagem..."):
            # Pré-processa para melhorar detecção
            try:
                import cv2
                from PIL import Image
                import io as _io

                pil = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
                arr = np.array(pil)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                # Aumenta contraste e nitidez
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                bgr_enhanced = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

                # Unsharp mask (nitidez)
                blur = cv2.GaussianBlur(bgr_enhanced, (0,0), 3)
                bgr_sharp = cv2.addWeighted(bgr_enhanced, 1.7, blur, -0.7, 0)

                # Salva como bytes
                _, enc = cv2.imencode(".jpg", bgr_sharp, [cv2.IMWRITE_JPEG_QUALITY, 97])
                img_bytes_proc = enc.tobytes()
            except:
                img_bytes_proc = img_bytes

            detected = detector.detect_cards_from_image(img_bytes_proc)

        hole_d  = detected.get("hole_cards", [])
        board_d = detected.get("board_cards", [])
        err_d   = detected.get("error")

        # ── Exibe o que foi detectado ──────────────────────────────────────
        st.markdown('<div class="box">', unsafe_allow_html=True)

        if err_d:
            st.error(f"⚠️ {err_d}")
        else:
            detected_ok = len(hole_d) >= 2

            dh1, dh2 = st.columns(2)
            with dh1:
                st.markdown("**Hole cards detectadas**")
                if hole_d:
                    st.markdown(cards_html(hole_d), unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#f44336;">❌ Não detectadas</span>',
                                unsafe_allow_html=True)
            with dh2:
                st.markdown("**Board detectado**")
                if board_d:
                    st.markdown(cards_html(board_d), unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#8b949e;">Preflop / vazio</span>',
                                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Permite correção rápida das cartas ────────────────────────────
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.markdown("**✏️ Corrija se necessário**")
        st.markdown('<div style="font-size:0.75rem;color:#8b949e;">Ex: Ah Ks</div>',
                    unsafe_allow_html=True)

        def card_val(lst, idx):
            return lst[idx] if len(lst) > idx else ""

        cc1, cc2 = st.columns(2)
        with cc1:
            h1 = st.text_input("Hole card 1", value=card_val(hole_d,0),
                                placeholder="Ah", key="ch1")
        with cc2:
            h2 = st.text_input("Hole card 2", value=card_val(hole_d,1),
                                placeholder="Ks", key="ch2")

        board_str_default = " ".join(board_d) if board_d else ""
        board_corr = st.text_input("Board (opcional)",
                                    value=board_str_default,
                                    placeholder="2h 7d Tc",
                                    key="cboard_corr")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔍 ANALISAR", key="cam_analyze"):
            final_hole  = parse_cards_string(f"{h1} {h2}")
            final_board = parse_cards_string(board_corr)
            if len(final_hole) >= 2:
                run_analysis(final_hole, final_board, pos_c, gt_c,
                             sit_c, rai_c, stk_c, pot_c)
            else:
                st.error("Corrija as hole cards antes de analisar.")

# ══════════════════════════════════════════════════════════════════════════════
# ABA SELETOR VISUAL DE CARTAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_pick:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    pg1, pg2 = st.columns(2)
    with pg1: gt_p = st.selectbox("Mesa", ["6max","9max"], key="pgt")
    with pg2:
        pos_list_p = positions(gt_p)
        pos_p = st.selectbox("Minha posição", pos_list_p, key="ppos")

    ps1, ps2 = st.columns(2)
    with ps1: stk_p = st.number_input("Stack (BB)", 1, 5000, 100, key="pstk")
    with ps2: pot_p = st.number_input("Pot (BB)",   0, 5000,   0, key="ppot")

    sit_p = st.selectbox("Situação", ["rfi","bb_defense","3bet"],
        format_func=lambda x:{"rfi":"Abrir pot","bb_defense":"BB vs raise","3bet":"3-bet"}[x],
        key="psit")
    rai_p = None
    if sit_p != "rfi":
        others_p = [p for p in pos_list_p if p != pos_p]
        rai_p = st.selectbox("Raiser", others_p, key="prai")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Seletor de cartas por grade ──────────────────────────────────────────
    RANKS_DISPLAY = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
    SUITS_DISPLAY = [("♠","s"),("♥","h"),("♦","d"),("♣","c")]

    if "picked" not in st.session_state:
        st.session_state.picked = []  # lista de cartas selecionadas

    MAX_CARDS = 7  # 2 hole + 5 board

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown(f"**Selecione suas cartas** ({len(st.session_state.picked)}/{MAX_CARDS})")
    st.markdown('<div style="font-size:0.75rem;color:#8b949e;">Primeiras 2 = hole cards • Demais = board</div>',
                unsafe_allow_html=True)

    # Mostra cartas selecionadas
    if st.session_state.picked:
        picked_html = cards_html(st.session_state.picked)
        st.markdown(picked_html, unsafe_allow_html=True)

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("⬅️ Remover última", key="undo"):
                st.session_state.picked.pop()
                st.rerun()
        with p_col2:
            if st.button("🗑️ Limpar tudo", key="clear"):
                st.session_state.picked = []
                st.rerun()
    else:
        st.markdown('<span style="color:#8b949e;font-size:0.85rem;">Nenhuma carta selecionada</span>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Grade de seleção ────────────────────────────────────────────────────
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("**Escolha rank e naipe:**")

    if "sel_rank" not in st.session_state: st.session_state.sel_rank = None
    if "sel_suit" not in st.session_state: st.session_state.sel_suit = None

    # Linha de ranks
    r_cols = st.columns(len(RANKS_DISPLAY))
    for i, rank in enumerate(RANKS_DISPLAY):
        with r_cols[i]:
            btn_label = f"**{rank}**" if st.session_state.sel_rank == rank else rank
            if st.button(rank, key=f"r_{rank}",
                         help=f"Selecionar rank {rank}"):
                st.session_state.sel_rank = rank
                st.session_state.sel_suit = None
                st.rerun()

    # Se rank selecionado, mostra naipes
    if st.session_state.sel_rank:
        st.markdown(f'<div style="margin:8px 0;color:#00c853;">Rank: <b>{st.session_state.sel_rank}</b> — escolha o naipe:</div>',
                    unsafe_allow_html=True)
        s_cols = st.columns(4)
        for i, (sym, code) in enumerate(SUITS_DISPLAY):
            with s_cols[i]:
                card = f"{st.session_state.sel_rank}{code}"
                already = card in st.session_state.picked
                label = f"{sym}" if not already else f"~~{sym}~~"
                color = "red" if code in ("h","d") else "white"
                if st.button(f"{sym}", key=f"s_{code}",
                             disabled=already or len(st.session_state.picked) >= MAX_CARDS,
                             help=card):
                    st.session_state.picked.append(card)
                    st.session_state.sel_rank = None
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Analisar ─────────────────────────────────────────────────────────────
    if len(st.session_state.picked) >= 2:
        if st.button("🔍 ANALISAR MÃO", key="pick_analyze"):
            h_p = st.session_state.picked[:2]
            b_p = st.session_state.picked[2:]
            run_analysis(h_p, b_p, pos_p, gt_p, sit_p, rai_p, stk_p, pot_p)

# ══════════════════════════════════════════════════════════════════════════════
# ABA POT ODDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_odds:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("**📊 Calculadora de Pot Odds**")

    oc1, oc2 = st.columns(2)
    with oc1: bet_o = st.number_input("Aposta a pagar (BB)", 0.5, 9999.0, 10.0, 0.5, key="obet")
    with oc2: pot_o = st.number_input("Pot atual (BB)",      1.0, 9999.0, 30.0, 1.0, key="opot")
    outs_o = st.number_input("Meus outs (0 = não sei)", 0, 20, 0, key="oouts")
    st.markdown('</div>', unsafe_allow_html=True)

    if bet_o > 0 and pot_o > 0:
        odds = engine.get_pot_odds(bet_o, pot_o)

        oc3, oc4 = st.columns(2)
        with oc3:
            st.markdown(f'<div class="sbox"><div class="sv">{odds["required_equity_pct"]}%</div>'
                        f'<div class="sl">Equity mínima</div></div>', unsafe_allow_html=True)
        with oc4:
            ratio = round(pot_o/bet_o, 1)
            st.markdown(f'<div class="sbox"><div class="sv">{ratio}:1</div>'
                        f'<div class="sl">Pot odds</div></div>', unsafe_allow_html=True)

        d = odds["decision"]
        cls2 = "res res-green" if "CALL" in d else ("res res-yel" if "DEPENDE" in d else "res res-red")
        st.markdown(f'<div class="{cls2}"><div class="res-lbl" style="font-size:1.1rem;">{d}</div></div>',
                    unsafe_allow_html=True)

        if outs_o > 0:
            eq_f = min(outs_o*4, 100)
            eq_t = min(outs_o*2, 100)
            req  = odds["required_equity_pct"]

            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.markdown(f"**Regra 4/2 com {outs_o} outs:**")
            a1, a2 = st.columns(2)
            with a1:
                clr = "#00c853" if eq_f >= req else "#f44336"
                st.markdown(f'<div class="sbox"><div class="sv" style="color:{clr};">{eq_f}%</div>'
                            f'<div class="sl">No Flop (×4)</div></div>', unsafe_allow_html=True)
            with a2:
                clr = "#00c853" if eq_t >= req else "#f44336"
                st.markdown(f'<div class="sbox"><div class="sv" style="color:{clr};">{eq_t}%</div>'
                            f'<div class="sl">No Turn (×2)</div></div>', unsafe_allow_html=True)

            if eq_f >= req:
                st.success(f"✅ Com {outs_o} outs você tem equity suficiente para chamar!")
            else:
                st.error(f"❌ Precisa de ≥ {round(req/4)} outs para justificar o call")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📖 Outs de referência"):
            st.markdown("""
| Outs | Draw | Equity Flop |
|------|------|------------|
| 15 | Flush + OESD | ~60% |
| 9 | Flush draw | ~36% |
| 8 | OESD | ~32% |
| 6 | Dois overcards | ~24% |
| 4 | Gutshot | ~16% |
| 2 | Pocket pair → set | ~8% |
            """)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:16px 0 6px;color:#21262d;font-size:0.65rem;">
  GTO Poker Advisor • PokerStars Edition • Uso educacional
</div>""", unsafe_allow_html=True)
