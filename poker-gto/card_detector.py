"""
card_detector.py
Detecção de cartas, posição, stack e pot via OpenCV + OCR (Tesseract).
Otimizado para layout PokerStars 6-max (desktop/mobile).
"""
import cv2
import numpy as np
from PIL import Image
import io
import re

try:
    import pytesseract
    # Windows: ajuste o caminho se necessário
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Aviso: pytesseract não instalado. OCR desativado.")

# ─── CONSTANTES ──────────────────────────────────────────────────────────────
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["s", "h", "d", "c"]

# Cores HSV típicas da PokerStars
HSV_POKERSTARS = {
    "red": [(0, 100, 100), (12, 255, 255), (165, 100, 100), (180, 255, 255)],
    "black": [(0, 0, 0), (180, 60, 90)],
    "green_table": [(35, 40, 40), (85, 255, 180)],
}


class CardDetector:
    def __init__(self):
        self.last_frame = None
        self.debug_mode = False

    # ─── PREPROCESSAMENTO ────────────────────────────────────────────────────
    def preprocess_image(self, image_bytes):
        """Converte bytes de imagem (print) para array OpenCV BGR."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_rgb = img.convert("RGB")
            img_array = np.array(img_rgb)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            self.last_frame = img_bgr
            return img_bgr
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            return None

    # ─── DETECÇÃO DE CARTAS ──────────────────────────────────────────────────
    def detect_cards_from_image(self, image_bytes):
        """Pipeline principal: detecta hole cards e board cards."""
        img = self.preprocess_image(image_bytes)
        if img is None:
            return {"hole_cards": [], "board_cards": [], "error": "Falha ao carregar imagem"}

        h, w = img.shape[:2]

        # Regiões ajustadas para layout PokerStars 6-max
        hole_region = img[int(h * 0.60):int(h * 0.85), int(w * 0.30):int(w * 0.70)]
        board_region = img[int(h * 0.25):int(h * 0.55), int(w * 0.20):int(w * 0.80)]

        hole_cards = self._extract_cards_from_region(hole_region, max_cards=2)
        board_cards = self._extract_cards_from_region(board_region, max_cards=5)

        return {
            "hole_cards": hole_cards,
            "board_cards": board_cards,
            "error": None
        }

    def _extract_cards_from_region(self, region, max_cards=5):
        """Extrai cartas de uma região usando threshold adaptativo + contornos."""
        cards = []
        if region is None or region.size == 0:
            return cards

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold adaptativo funciona melhor em interfaces digitais
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morfologia para unir bordas das cartas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_regions = []
        h, w = region.shape[:2]
        min_area = (h * w) * 0.01  # 1% da área da região

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = cw / ch if ch > 0 else 0

            # Cartas de poker têm proporção ~0.7 (largura/altura)
            if area > min_area and 0.5 < aspect < 0.85:
                card_regions.append((x, y, cw, ch))

        # Ordena da esquerda para direita
        card_regions = sorted(card_regions, key=lambda r: r[0])
        card_regions = card_regions[:max_cards]

        for (x, y, cw, ch) in card_regions:
            margin = 3
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + cw + margin)
            y2 = min(h, y + ch + margin)
            
            card_img = region[y1:y2, x1:x2]
            card = self._read_card(card_img)
            if card:
                cards.append(card)

        return cards

    def _read_card(self, card_img):
        """Lê rank e naipe de uma carta."""
        if card_img is None or card_img.size == 0:
            return None

        h, w = card_img.shape[:2]
        if h < 20 or w < 15:
            return None

        # Região do rank (canto superior esquerdo)
        rank_region = card_img[0:int(h * 0.40), 0:int(w * 0.50)]
        # Região do naipe (logo abaixo do rank)
        suit_region = card_img[int(h * 0.35):int(h * 0.75), 0:int(w * 0.50)]

        rank = self._detect_rank(rank_region)
        suit = self._detect_suit(suit_region)

        if rank and suit:
            return f"{rank}{suit}"
        return None

    def _detect_rank(self, region):
        """Detecta rank via OCR (Tesseract) com fallback heurístico."""
        if region is None or region.size == 0:
            return None

        if HAS_TESSERACT:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=AKQJT23456789'
            text = pytesseract.image_to_string(thresh, config=custom_config).strip()
            text = text.replace("10", "T")
            
            if text in RANKS:
                return text

        # Fallback heurístico
        return self._fallback_rank_heuristic(region)

    def _fallback_rank_heuristic(self, region):
        """Heurística de fallback baseada em contornos."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, cw, ch = cv2.boundingRect(largest)
        aspect = cw / ch if ch > 0 else 0
        fill = area / (cw * ch) if (cw * ch) > 0 else 0

        epsilon = 0.04 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        corners = len(approx)

        if fill > 0.85 and aspect > 0.7:
            return "A"
        elif fill > 0.75 and corners <= 5:
            return "T"
        elif fill > 0.70:
            return "K"
        elif fill > 0.60 and corners > 6:
            return "Q"
        elif area > 300:
            return "K"
        elif area > 200:
            return "J"
        return None

    def _detect_suit(self, region):
        """Detecta naipe pela cor (PokerStars: vermelho=♥♦, preto=♠♣)."""
        if region is None or region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Máscara vermelha (hearts/diamonds)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Máscara escura (spades/clubs)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 90]))

        red_count = cv2.countNonZero(red_mask)
        dark_count = cv2.countNonZero(dark_mask)

        if red_count > dark_count:
            # Vermelho: distingue hearts (arredondado) de diamonds (losango)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / ch if ch > 0 else 1
                if aspect > 0.9:  # losango é mais largo/quadrado
                    return "d"
            return "h"
        else:
            # Escuro: distingue spades (pontiagudo) de clubs (3 bolinhas)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / ch if ch > 0 else 1
                if aspect < 0.8:  # spades tende a ser mais alto
                    return "s"
            return "c"

    # ─── OCR PARA NÚMEROS (Stack, Pot, Bet) ─────────────────────────────────
    def _extract_text_value(self, region):
        """Extrai valores numéricos (suporta formato brasileiro: 5,67)."""
        if not HAS_TESSERACT or region is None or region.size == 0:
            return None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Whitelist: números, vírgula, ponto, B
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,B'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()
        
        # Remove "BB" e espaços, mantém apenas número
        cleaned = re.sub(r'[^\d.,]', '', text)
        cleaned = cleaned.replace(',', '.')  # Formato brasileiro → padrão
        
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    # ─── DETECÇÃO DE POSIÇÃO ─────────────────────────────────────────────────
    def detect_position(self, img):
        """
        Detecta posição do jogador ativo.
        Heurística: procura indicador de 'ativo' (borda amarela/verde) nas regiões de assentos.
        Layout 6-max PokerStars (ajuste conforme necessário).
        """
        h, w = img.shape[:2]
        
        # Regiões aproximadas dos 6 assentos (y1, y2, x1, x2)
        positions = {
            "HERO": (int(h * 0.75), h, int(w * 0.35), int(w * 0.65)),
            "BTN": (int(h * 0.60), int(h * 0.75), int(w * 0.75), w),
            "SB": (int(h * 0.40), int(h * 0.55), int(w * 0.75), w),
            "BB": (int(h * 0.25), int(h * 0.40), int(w * 0.60), int(w * 0.80)),
            "UTG": (int(h * 0.25), int(h * 0.40), int(w * 0.20), int(w * 0.40)),
            "MP": (int(h * 0.40), int(h * 0.55), 0, int(w * 0.25)),
        }

        for pos_name, (y1, y2, x1, x2) in positions.items():
            region = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            # Procura amarelo/verde (indicador de vez/ativo na PokerStars)
            active_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([90, 255, 255]))
            if cv2.countNonZero(active_mask) > 50:
                return pos_name
                
        return "HERO"

    # ─── EXTRAÇÃO COMPLETA DE INFORMAÇÕES ────────────────────────────────────
    def extract_game_info(self, image_bytes):
        """
        Extrai todas as informações para comparação com tabelas GTO.
        Retorna dict estruturado.
        """
        img = self.preprocess_image(image_bytes)
        if img is None:
            return {"error": "Falha ao carregar imagem"}

        h, w = img.shape[:2]

        # Regiões ajustadas para layout PokerStars (proporções)
        regions = {
            "hole_cards": img[int(h * 0.60):int(h * 0.85), int(w * 0.30):int(w * 0.70)],
            "board": img[int(h * 0.25):int(h * 0.55), int(w * 0.20):int(w * 0.80)],
            "pot": img[int(h * 0.20):int(h * 0.28), int(w * 0.38):int(w * 0.62)],
            "stack": img[int(h * 0.80):int(h * 0.92), int(w * 0.35):int(w * 0.65)],
            "bet": img[int(h * 0.85):int(h * 0.95), int(w * 0.70):int(w * 0.95)],
        }

        # 1. Cartas
        hole_cards_raw = self._extract_cards_from_region(regions["hole_cards"], max_cards=2)
        board_cards_raw = self._extract_cards_from_region(regions["board"], max_cards=5)

        # 2. Normaliza hole cards para formato GTO
        normalized_hand = None
        if len(hole_cards_raw) == 2:
            normalized_hand = self.normalize_hand(hole_cards_raw[0], hole_cards_raw[1])

        # 3. Valores numéricos (Stack em BB, Pot, Bet)
        stack_bb = self._extract_text_value(regions["stack"])
        pot = self._extract_text_value(regions["pot"])
        bet_amount = self._extract_text_value(regions["bet"])

        # 4. Posição
        position = self.detect_position(img)

        return {
            "hole_cards_raw": hole_cards_raw,
            "hole_cards_gto": normalized_hand,
            "board_cards": board_cards_raw,
            "pot": pot,
            "stack_bb": stack_bb,
            "bet_amount": bet_amount,
            "position": position,
            "error": None
        }

    # ── NORMALIZAÇÃO PARA GTO ───────────────────────────────────────────────
    def normalize_hand(self, card1, card2):
        """
        Normaliza 2 cartas para formato de tabela GTO.
        Ex: Ah Ks → AKs, Ah Kd → AKo, Ah As → AA
        """
        if not card1 or not card2 or len(card1) < 2 or len(card2) < 2:
            return None

        r1, s1 = card1[:-1], card1[-1]
        r2, s2 = card2[:-1], card2[-1]

        r1 = r1.replace("10", "T").upper()
        r2 = r2.replace("10", "T").upper()

        if r1 not in RANKS or r2 not in RANKS:
            return None

        idx1 = RANKS.index(r1)
        idx2 = RANKS.index(r2)

        # Ordena por rank (maior primeiro)
        if idx1 > idx2:
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        if r1 == r2:
            return f"{r1}{r2}"  # Par
        elif s1 == s2:
            return f"{r1}{r2}s"  # Suited
        else:
            return f"{r1}{r2}o"  # Offsuit


# ─── FUNÇÕES AUXILIARES ──────────────────────────────────────────────────────
def manual_card_input_to_normalized(cards_str):
    """Converte string de cartas manual para formato normalizado."""
    detector = CardDetector()
    cards = cards_str.strip().split()
    if len(cards) >= 2:
        return detector.normalize_hand(cards[0], cards[1])
    return None


def parse_cards_string(cards_str):
    """Parseia string como 'Ah Kd 2c 5h 9s' → ['Ah', 'Kd', '2c', '5h', '9s']"""
    if not cards_str:
        return []
    
    cards_str = cards_str.replace("10", "T")
    pattern = r'([AKQJTakqjt2-9]{1,2}[shdcSHDC])'
    found = re.findall(pattern, cards_str)

    result = []
    for card in found:
        rank = card[:-1].upper()
        suit = card[-1].lower()
        if rank in RANKS and suit in SUITS:
            result.append(f"{rank}{suit}")
    return result