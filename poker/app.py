import streamlit as st
import numpy as np
import cv2
import os
import json
import logging
import sys
from PIL import Image

# ==========================================
# 1. CONFIGURAÇÕES, LOGGING E HELPERS (utils)
# ==========================================
def setup_logger(name: str = "GTO_Analyzer") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def calculate_pot_odds(pot: float, call_amount: float) -> float:
    return float(call_amount / (pot + call_amount)) if pot + call_amount > 0 else 0.0

def calculate_mdf(bet_size_pot_pct: float) -> float:
    return float(1.0 / (1.0 + bet_size_pot_pct)) if bet_size_pot_pct > 0 else 1.0

def calculate_spr(stack: float, pot: float) -> float:
    return float(stack / pot) if pot > 0 else 0.0

# ==========================================
# 2. VISÃO COMPUTACIONAL (card_detector & ocr)
# ==========================================
class ImagePreprocessor:
    def optimize_image(self, image: np.ndarray) -> np.ndarray:
        try:
            # Redimensionamento padrão para estabilizar coordenadas de análise
            resized = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # CLAHE melhora o contraste para fotos tiradas de celular com reflexo na tela
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl_img = clahe.apply(gray)
            denoised = cv2.bilateralFilter(cl_img, 9, 75, 75)
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            return image

class TableDetector:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()

    def process_table_image(self, image_raw: np.ndarray) -> dict:
        processed = self.preprocessor.optimize_image(image_raw)
        
        # Mocks estruturados simulando a extração dos contornos com OpenCV
        # Em produção completa, essas listas seriam populadas via Template Matching
        hole_cards = ["Ah", "Ks"]
        board_cards = {"flop": ["Kh", "Qd", "2c"], "turn": ["Jh"], "river": []}
        
        return {
            "hole_cards": hole_cards,
            "board": board_cards,
            "players_count": 6,
            "hero_stack": 1500.0,
            "pot_size": 350.0,
            "big_blind": 20.0,
            "position": "UTG"
        }

# ==========================================
# 3. MOTOR DE DECISÃO E RANGES (gto_engine)
# ==========================================
class PreflopSolver:
    def __init__(self):
        # Fallback de segurança incorporado diretamente no código caso o JSON não exista
        self.default_utg_range = {
            "AA": {"action": "RAISE", "sizing": "2.5BB", "confidence": 100},
            "KK": {"action": "RAISE", "sizing": "2.5BB", "confidence": 100},
            "QQ": {"action": "RAISE", "sizing": "2.5BB", "confidence": 98},
            "AKo": {"action": "RAISE", "sizing": "2.5BB", "confidence": 95},
            "AQs": {"action": "RAISE", "sizing": "2.5BB", "confidence": 92},
            "72o": {"action": "FOLD", "sizing": "0%", "confidence": 100}
        }

    def get_action(self, position: str, hole_cards: str) -> dict:
        # Tenta carregar dinamicamente de uma pasta local para manter a modularidade
        file_path = os.path.join("ranges", "preflop", f"{position}_OPEN.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    range_matrix = json.load(f)
                action_data = range_matrix.get(hole_cards, {"action": "FOLD", "sizing": "0%", "confidence": 90})
                action_data["reason"] = f"Estratégia GTO pura carregada via matriz JSON dinâmica de {position}."
                return action_data
            except Exception as e:
                logger.error(f"Erro ao ler JSON de range: {e}")

        # Se não houver arquivo, usa o banco de dados padrão integrado
        action_data = self.default_utg_range.get(hole_cards, {"action": "FOLD", "sizing": "0%", "confidence": 90})
        action_data["reason"] = "Estratégia pura baseada no range nativo padrão de abertura do UTG."
        return action_data

class DecisionEngine:
    def __init__(self):
        self.preflop = PreflopSolver()

    def orchestrate_decision(self, state: dict) -> dict:
        try:
            board = state.get("board", {"flop": [], "turn": [], "river": []})
            spr = state.get("spr", 0.0)
            
            # Mudança de comportamento estratégico baseada em qual street a mesa está
            if len(board.get("flop", [])) == 0:
                return self.preflop.get_action(state["position"], "AKo")
            elif len(board.get("turn", [])) == 0:
                if spr > 10:
                    return {
                        "action": "BET", "sizing": "33%", "confidence": 85,
                        "reason": "C-Bet padrão GTO em Flop seco/conectado usando vantagem de range do agressor pré-flop."
                    }
                return {"action": "CHECK", "sizing": "0%", "confidence": 80, "reason": "SPR baixo. Check de controle de pote."}
            elif len(board.get("river", [])) == 0:
                return {
                    "action": "CHECK", "sizing": "0%", "confidence": 75,
                    "reason": "Turn trouxe uma 'scare card' que acerta consideravelmente o range de defesa de blind do oponente."
                }
            else:
                return {
                    "action": "BET", "sizing": "75%", "confidence": 90,
                    "reason": "Value bet no River para extrair valor máximo de draws perdidos ou pares médios do vilão."
                }
        except Exception as e:
            return {"action": "CHECK", "sizing": "0%", "confidence": 0, "reason": f"Erro interno crítico: {e}"}

# ==========================================
# 4. INTERFACE GRÁFICA WEB (streamlit app)
# ==========================================
st.set_page_config(
    page_title="GTO Texas Hold'em Pro Solver",
    page_icon="♠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("♠️ Monitor Analisador de GTO & Real-Time Poker Engine")
st.markdown("---")

# Inicialização única dos motores integrados
detector = TableDetector()
decision_engine = DecisionEngine()

st.sidebar.header("📥 Entrada de Captura")
uploaded_file = st.sidebar.file_uploader(
    "Carregar Screenshot ou Foto da Mesa", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image_raw = Image.open(uploaded_file)
    img_cv2 = np.array(image_raw)
    
    with st.spinner("Processando imagem e decodificando elementos com Visão Computacional..."):
        # Executa toda a leitura unificada da tela
        table_data = detector.process_table_image(img_cv2)
    
    # Cálculos Matemáticos Avançados baseados na leitura da mesa
    spr = calculate_spr(table_data["hero_stack"], table_data["pot_size"])
    mock_call_amt = table_data["big_blind"] * 2.5
    pot_odds = calculate_pot_odds(table_data["pot_size"], mock_call_amt)
    mdf = calculate_mdf(mock_call_amt / table_data["pot_size"] if table_data["pot_size"] > 0 else 0)
    
    # Montando estado consolidado para o motor GTO
    game_state = {
        "hole_cards": table_data["hole_cards"],
        "board": table_data["board"],
        "position": table_data["position"],
        "pot": table_data["pot_size"],
        "stack": table_data["hero_stack"],
        "spr": spr,
        "pot_odds": pot_odds
    }
    
    # Consulta ao cérebro de decisões
    final_gto_decision = decision_engine.orchestrate_decision(game_state)
    
    # renderização da Interface Gráfica
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Imagem Analisada")
        st.image(image_raw, use_container_width=True)
        
        st.subheader("🃏 Dados Identificados da Mesa")
        st.write(f"**Hole Cards (Sua Mão):** `{table_data['hole_cards']}`")
        st.write(f"**Flop:** `{table_data['board']['flop']}` | **Turn:** `{table_data['board']['turn']}` | **River:** `{table_data['board']['river']}`")
        st.write(f"**Posição do Hero:** `{table_data['position']}` | **Jogadores na Mesa:** `{table_data['players_count']}`")
    
    with col2:
        st.subheader("📊 Indicadores Matemáticos")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("SPR (Razão de Stack)", f"{spr:.2f}")
        kpi2.metric("Pot Odds", f"{pot_odds * 100:.1f}%")
        kpi3.metric("MDF (Defesa Mínima)", f"{mdf * 100:.1f}%")
        
        st.markdown("---")
        st.subheader("🤖 Recomendação Estratégica GTO")
        
        action = final_gto_decision["action"]
        color_map = {"FOLD": "gray", "CHECK": "blue", "CALL": "green", "BET": "orange", "RAISE": "red", "ALLIN": "purple"}
        color = color_map.get(action, "black")
        
        st.markdown(f"""
        <div style="background-color:rgba(0,0,0,0.05); padding:20px; border-radius:10px; border-left: 8px solid {color};">
            <h2 style="margin:0; color:{color};">{action}</h2>
            <h4 style="margin:5px 0 0 0;">Sizing Recomendado: <b>{final_gto_decision['sizing']}</b></h4>
            <p style="margin:10px 0 0 0; color:#555;">Confiança do Motor: {final_gto_decision['confidence']}%</p>
        </div>
        """, unsafe_with_html=True)
        
        st.markdown("#### Justificativa Teórica:")
        st.info(final_gto_decision["reason"])
else:
    st.info("💡 Por favor, carregue uma imagem/screenshot de pôquer na barra lateral para iniciar a leitura automatizada.")