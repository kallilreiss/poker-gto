"""
Poker Vision GTO — Streamlit Frontend
Analise screenshots de poker e receba recomendações GTO automáticas.
"""

import io
import sys
import os
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from card_detector.detect_hole_cards import detect_hole_cards
from card_detector.detect_board import detect_board
from table_detector.detect_stacks import detect_stacks
from table_detector.detect_pot import detect_pot
from table_detector.detect_positions import detect_position
from table_detector.detect_actions import detect_actions
from table_detector.detect_players import detect_num_players
from table_detector.detect_street import detect_street
from gto_engine.lookup import get_gto_action
from gto_engine.strategy_engine import categorize_hand

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Poker Vision GTO",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --green: #1b5e20;
    --gold:  #ffd600;
    --card-bg: #1e1e2e;
    --text:  #e0e0e0;
  }
  .main { background: #0d1117; }
  .stApp { background: #0d1117; color: var(--text); }

  .gto-card {
    background: var(--card-bg);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
  }
  .gto-card h4 { color: var(--gold); margin-bottom: 8px; font-size: 0.85rem; letter-spacing: 1px; }
  .gto-card .value { font-size: 1.4rem; font-weight: 700; color: #fff; }
  .gto-card .sub   { font-size: 0.8rem; color: #888; margin-top: 4px; }

  .action-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: 1px;
    margin-top: 4px;
  }
  .action-CHECK  { background: #1565c0; color: #fff; }
  .action-BET    { background: #e65100; color: #fff; }
  .action-RAISE  { background: #b71c1c; color: #fff; }
  .action-CALL   { background: #2e7d32; color: #fff; }
  .action-FOLD   { background: #37474f; color: #ccc; }
  .action-ALL_IN { background: #4a148c; color: #fff; }

  .freq-bar-wrap { margin: 6px 0; }
  .freq-label    { font-size: 0.75rem; color: #aaa; margin-bottom: 2px; }
  .freq-bar      { height: 14px; border-radius: 7px; }

  .card-pip {
    display: inline-block;
    background: #fff;
    color: #111;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 1.2rem;
    font-weight: 800;
    margin: 2px;
    min-width: 38px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
  }
  .card-red { color: #c62828; }

  .section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: #ffd600;
    text-transform: uppercase;
    margin: 18px 0 8px;
  }
  .justification-box {
    background: #161b22;
    border-left: 3px solid #ffd600;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #ccc;
    line-height: 1.6;
  }
  div[data-testid="stSidebar"] { background: #161b22; }
</style>
""", unsafe_allow_html=True)

SUIT_SYMBOLS = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
RED_SUITS    = {"h", "d"}
FREQ_COLORS  = {
    "CHECK": "#1565c0", "BET": "#e65100", "RAISE": "#b71c1c",
    "CALL": "#2e7d32", "FOLD": "#37474f", "ALL_IN": "#4a148c",
}

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_card(card: str) -> str:
    if len(card) < 2:
        return card
    rank, suit = card[0].upper(), card[1].lower()
    symbol = SUIT_SYMBOLS.get(suit, suit)
    css_class = "card-red" if suit in RED_SUITS else ""
    return f'<span class="card-pip {css_class}">{rank}{symbol}</span>'


def render_cards(cards: list[str]) -> str:
    return " ".join(render_card(c) for c in cards) if cards else "<em style='color:#555'>—</em>"


def freq_bar(action: str, pct: int) -> str:
    color = FREQ_COLORS.get(action, "#888")
    return (
        f'<div class="freq-bar-wrap">'
        f'<div class="freq-label">{action} — {pct}%</div>'
        f'<div class="freq-bar" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
    )


def analyze_image(pil_img: Image.Image) -> dict:
    image = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    hero_cards     = detect_hole_cards(image)
    board          = detect_board(image)
    stacks         = detect_stacks(image)
    pot_bb         = detect_pot(image)
    num_players    = detect_num_players(image)
    position       = detect_position(image, num_players)
    street         = detect_street(image)
    avail_actions  = detect_actions(image)

    hero_stack     = stacks.get("hero_stack", 0.0)
    villain_stacks = stacks.get("villains", [])

    gto = get_gto_action(
        hero_cards=hero_cards,
        board=board,
        position=position,
        street=street,
        pot_bb=pot_bb,
        hero_stack=hero_stack,
        villain_stacks=villain_stacks,
    )

    hand_cat = ""
    if hero_cards:
        try:
            hand_cat = categorize_hand(hero_cards, board).value
        except Exception:
            hand_cat = ""

    return {
        "hero_cards":      hero_cards,
        "board":           board,
        "position":        position,
        "street":          street,
        "pot_bb":          pot_bb,
        "hero_stack":      hero_stack,
        "villain_stacks":  villain_stacks,
        "available_actions": avail_actions,
        "gto_action":      gto.get("action", "CHECK"),
        "gto_frequencies": gto.get("frequencies", {}),
        "gto_bet_size":    gto.get("bet_size"),
        "justification":   gto.get("justification", ""),
        "hand_category":   hand_cat,
        "timestamp":       datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ♠ Poker Vision GTO")
    st.markdown("---")
    st.markdown("### Como usar")
    st.markdown(
        "1. Faça upload de uma screenshot da mesa\n"
        "2. Clique em **Analisar Mão**\n"
        "3. Veja a recomendação GTO\n"
        "4. Consulte o histórico de análises"
    )
    st.markdown("---")
    st.markdown("### Posições suportadas")
    st.markdown("BTN · CO · HJ · MP · UTG · UTG+1 · SB · BB")
    st.markdown("### Mesas suportadas")
    st.markdown("6-max · 9-max")
    st.markdown("---")

    if st.session_state.history:
        st.markdown(f"**Análises realizadas:** {len(st.session_state.history)}")
        if st.button("🗑 Limpar histórico"):
            st.session_state.history = []
            st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# ♠ Poker Vision GTO")
st.markdown("*Upload de screenshot → Análise automática → Recomendação GTO*")
st.markdown("---")

tab_analyze, tab_history = st.tabs(["📸 Análise", "📋 Histórico"])

# ─── Tab: Análise ────────────────────────────────────────────────────────────
with tab_analyze:
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<p class="section-title">Upload da screenshot</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Arraste ou clique para selecionar",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded)
            with col_preview:
                st.markdown('<p class="section-title">Preview</p>', unsafe_allow_html=True)
                st.image(pil_img, use_column_width=True, caption="Screenshot carregada")

            st.markdown("")
            analyze_btn = st.button("🔍 Analisar Mão", type="primary", use_container_width=True)

            if analyze_btn:
                with st.spinner("Analisando imagem…"):
                    try:
                        result = analyze_image(pil_img)
                        st.session_state.last_result = result
                        st.session_state.history.insert(0, result)
                    except Exception as e:
                        st.error(f"Erro na análise: {e}")
                        result = None

    # ── Results ───────────────────────────────────────────────────────────────
    if "last_result" in st.session_state and uploaded:
        r = st.session_state.last_result
        st.markdown("---")
        st.markdown('<p class="section-title">Resultado da análise</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f'<div class="gto-card"><h4>CARTAS DO HERO</h4>'
                f'<div class="value">{render_cards(r["hero_cards"]) if r["hero_cards"] else "—"}</div>'
                f'<div class="sub">{r["hand_category"] or "—"}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="gto-card"><h4>BOARD</h4>'
                f'<div class="value">{render_cards(r["board"]) if r["board"] else "Preflop"}</div>'
                f'<div class="sub">Street: {r["street"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="gto-card"><h4>POSIÇÃO / STACKS</h4>'
                f'<div class="value">{r["position"]}</div>'
                f'<div class="sub">Hero: {r["hero_stack"]:.1f} BB &nbsp;|&nbsp; Pot: {r["pot_bb"]:.2f} BB</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            action = r["gto_action"]
            size_txt = f" {r['gto_bet_size']}" if r.get("gto_bet_size") else ""
            st.markdown(
                f'<div class="gto-card"><h4>AÇÃO GTO</h4>'
                f'<span class="action-badge action-{action}">{action}{size_txt}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Frequency bars
        st.markdown('<p class="section-title">Frequências GTO</p>', unsafe_allow_html=True)
        freq_html = "".join(
            freq_bar(a, p) for a, p in sorted(r["gto_frequencies"].items(), key=lambda x: -x[1])
        )
        st.markdown(f'<div class="gto-card">{freq_html}</div>', unsafe_allow_html=True)

        # Justification
        st.markdown('<p class="section-title">Justificativa estratégica</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="justification-box">{r["justification"]}</div>',
            unsafe_allow_html=True,
        )

        # Villain stacks
        if r["villain_stacks"]:
            st.markdown('<p class="section-title">Stacks dos vilões</p>', unsafe_allow_html=True)
            df_v = pd.DataFrame(
                {"Jogador": [f"Vilão {i+1}" for i in range(len(r["villain_stacks"]))],
                 "Stack (BB)": r["villain_stacks"]}
            )
            st.dataframe(df_v, use_container_width=True, hide_index=True)

        # Export single result
        st.download_button(
            "⬇ Exportar esta análise (JSON)",
            data=json.dumps(r, ensure_ascii=False, indent=2),
            file_name=f"gto_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ─── Tab: Histórico ──────────────────────────────────────────────────────────
with tab_history:
    st.markdown('<p class="section-title">Histórico de análises</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("Nenhuma análise realizada ainda. Faça o upload de uma screenshot para começar.")
    else:
        rows = []
        for i, r in enumerate(st.session_state.history):
            rows.append({
                "#": i + 1,
                "Horário":    r.get("timestamp", "—"),
                "Hero":       " ".join(r.get("hero_cards", [])) or "—",
                "Board":      " ".join(r.get("board", [])) or "—",
                "Posição":    r.get("position", "—"),
                "Street":     r.get("street", "—"),
                "Pot (BB)":   r.get("pot_bb", 0),
                "Stack (BB)": r.get("hero_stack", 0),
                "Categoria":  r.get("hand_category", "—"),
                "Ação GTO":   r.get("gto_action", "—"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Exportar histórico (CSV)",
            data=csv,
            file_name=f"gto_historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Detail expanders
        st.markdown("---")
        st.markdown('<p class="section-title">Detalhe por análise</p>', unsafe_allow_html=True)
        for i, r in enumerate(st.session_state.history[:10]):
            with st.expander(
                f"#{i+1} — {r.get('timestamp','—')}  |  "
                f"{' '.join(r.get('hero_cards',[]))} vs Board {' '.join(r.get('board',[]))}  |  "
                f"Ação: **{r.get('gto_action','—')}**"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.json({k: v for k, v in r.items() if k != "justification"})
                with col_b:
                    st.markdown(f"**Justificativa:**\n\n{r.get('justification','')}")
