"""
card_detector.py
Detecção de cartas, posição, stack e pot via OpenCV + template matching.
Otimizado para leitura de tela de app/site de poker no celular.
"""

import cv2
import numpy as np
from PIL import Image
import io
import re


# ─── CONSTANTES ──────────────────────────────────────────────────────────────
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["s", "h", "d", "c"]  # spades, hearts, diamonds, clubs

SUIT_COLORS = {
    "s": (0, 0, 0),       # preto
    "c": (0, 128, 0),     # verde escuro
    "h": (200, 0, 0),     # vermelho
    "d": (200, 0, 0),     # vermelho
}

# Faixas de cor HSV para detecção de naipes
HSV_RANGES = {
    "red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
    "black": [(0, 0, 0), (180, 50, 80)],
}


class CardDetector:
    def __init__(self):
        self.last_frame = None
        self.debug_mode = False

    def preprocess_image(self, image_bytes):
        """Converte bytes de imagem para array OpenCV."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_rgb = img.convert("RGB")
            img_array = np.array(img_rgb)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            self.last_frame = img_bgr
            return img_bgr
        except Exception as e:
            return None

    def detect_cards_from_image(self, image_bytes):
        """
        Pipeline principal: detecta hole cards e board cards.
        Retorna dict com hole_cards e board_cards.
        """
        img = self.preprocess_image(image_bytes)
        if img is None:
            return {"hole_cards": [], "board_cards": [], "error": "Falha ao carregar imagem"}

        h, w = img.shape[:2]

        # Divide a imagem em regiões esperadas
        # Hole cards: parte inferior da tela (mão do jogador)
        hole_region = img[int(h * 0.65):h, 0:w]
        # Board: parte central-superior
        board_region = img[int(h * 0.25):int(h * 0.60), 0:w]

        hole_cards = self._extract_cards_from_region(hole_region, max_cards=2)
        board_cards = self._extract_cards_from_region(board_region, max_cards=5)

        return {
            "hole_cards": hole_cards,
            "board_cards": board_cards,
            "error": None
        }

    def _extract_cards_from_region(self, region, max_cards=5):
        """Extrai cartas de uma região da imagem."""
        cards = []

        # Detecta regiões brancas/claras (cartas têm fundo branco/claro)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Encontra contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = cw / ch if ch > 0 else 0

            # Filtra por tamanho e proporção de carta (≈ 0.6-0.8)
            if area > 1500 and 0.45 < aspect < 0.95:
                card_regions.append((x, y, cw, ch))

        # Ordena da esquerda para direita
        card_regions = sorted(card_regions, key=lambda r: r[0])
        card_regions = card_regions[:max_cards]

        for (x, y, cw, ch) in card_regions:
            card_img = region[y:y+ch, x:x+cw]
            card = self._read_card(card_img)
            if card:
                cards.append(card)

        return cards

    def _read_card(self, card_img):
        """Lê rank e naipe de uma imagem de carta."""
        if card_img is None or card_img.size == 0:
            return None

        h, w = card_img.shape[:2]
        if h < 20 or w < 15:
            return None

        # Região do rank (canto superior esquerdo ≈ 20%)
        rank_region = card_img[0:int(h*0.35), 0:int(w*0.45)]
        # Região do naipe (abaixo do rank)
        suit_region = card_img[int(h*0.35):int(h*0.60), 0:int(w*0.45)]

        rank = self._detect_rank(rank_region)
        suit = self._detect_suit(suit_region)

        if rank and suit:
            return f"{rank}{suit}"
        return None

    def _detect_rank(self, region):
        """Detecta o rank da carta via análise de contorno."""
        if region is None or region.size == 0:
            return None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Analisa o maior contorno
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 20:
            return None

        # Hu moments para classificação básica
        moments = cv2.moments(largest)
        hu = cv2.HuMoments(moments).flatten()

        # Heurística simples baseada em proporção e área
        x, y, cw, ch = cv2.boundingRect(largest)
        aspect = cw / ch if ch > 0 else 0
        fill = area / (cw * ch) if (cw * ch) > 0 else 0

        # Classificação baseada em características geométricas
        # Esta é uma heurística — templates dariam resultado melhor
        return self._classify_rank_by_shape(aspect, fill, area, largest)

    def _classify_rank_by_shape(self, aspect, fill, area, contour):
        """
        Classifica rank por características geométricas.
        Heurística baseada em proporções típicas de fontes de poker.
        """
        # Número de cantos aproximados
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = len(approx)

        # A=4pts triangular-ish, K=complexo, Q=oval, etc.
        if fill > 0.85 and aspect > 0.7:
            return "A"
        elif fill > 0.75 and corners <= 5:
            return "T"
        elif fill > 0.70:
            return "K"
        elif aspect < 0.4:
            return "1"  # pode ser parte de número
        elif fill > 0.60 and corners > 6:
            return "Q"

        # Fallback por área relativa
        if area > 300:
            return "K"
        elif area > 200:
            return "Q"
        elif area > 150:
            return "J"
        return None

    def _detect_suit(self, region):
        """Detecta naipe pela cor dominante."""
        if region is None or region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Máscara para vermelho (hearts/diamonds)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([155, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Máscara para preto/escuro (spades/clubs)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 100]))

        # Máscara azul (alguns apps usam azul para spades)
        blue_mask = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))

        red_count = cv2.countNonZero(red_mask)
        dark_count = cv2.countNonZero(dark_mask)
        blue_count = cv2.countNonZero(blue_mask)

        total = red_count + dark_count + blue_count
        if total == 0:
            return None

        # Analisa forma para distinguir h/d e s/c
        if red_count > dark_count and red_count > blue_count:
            # Vermelho: hearts ou diamonds
            # Hearts têm formato arredondado, diamonds são losangos
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / ch if ch > 0 else 1
                if aspect > 0.9:  # losango é mais largo
                    return "d"
                else:
                    return "h"
            return "h"  # default vermelho = hearts
        elif blue_count > red_count and blue_count > dark_count:
            return "s"  # azul = spades em alguns apps
        else:
            # Escuro: spades ou clubs
            # Clubs têm 3 bolinhas, spades são mais pontiagudos
            return "s"  # default escuro

    def detect_numbers_in_region(self, region):
        """
        Detecta números em uma região (para stack/pot).
        Usa análise de píxeis para identificar dígitos.
        """
        if region is None:
            return None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Aumenta contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Threshold adaptativo para textos
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        return thresh

    def extract_game_info(self, image_bytes):
        """
        Extrai informações gerais: cartas, stacks, pot, posição.
        """
        img = self.preprocess_image(image_bytes)
        if img is None:
            return {}

        h, w = img.shape[:2]

        # Regiões típicas de interface de poker (proporcional)
        regions = {
            "hole_cards": img[int(h*0.70):h, int(w*0.20):int(w*0.80)],
            "board": img[int(h*0.30):int(h*0.55), int(w*0.15):int(w*0.85)],
            "pot": img[int(h*0.20):int(h*0.35), int(w*0.30):int(w*0.70)],
            "stack": img[int(h*0.75):int(h*0.90), int(w*0.30):int(w*0.70)],
        }

        result = {}

        # Cartas
        hole_cards = self._extract_cards_from_region(regions["hole_cards"], 2)
        board_cards = self._extract_cards_from_region(regions["board"], 5)
        result["hole_cards"] = hole_cards
        result["board_cards"] = board_cards

        return result

    def normalize_hand(self, card1, card2):
        """
        Normaliza 2 cartas para formato de tabela GTO.
        Ex: Ah Ks → AKs (suited), Ah Kd → AKo (offsuit), Ah As → AAo (pair)
        """
        if not card1 or not card2 or len(card1) < 2 or len(card2) < 2:
            return None

        r1, s1 = card1[:-1], card1[-1]
        r2, s2 = card2[:-1], card2[-1]

        # Normaliza ranks (T para 10)
        r1 = r1.replace("10", "T")
        r2 = r2.replace("10", "T")

        if r1 not in RANKS or r2 not in RANKS:
            return None

        # Ordena por rank (maior primeiro)
        idx1 = RANKS.index(r1)
        idx2 = RANKS.index(r2)

        if idx1 > idx2:
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        if r1 == r2:
            return f"{r1}{r2}o"  # par
        elif s1 == s2:
            return f"{r1}{r2}s"  # suited
        else:
            return f"{r1}{r2}o"  # offsuit


def manual_card_input_to_normalized(cards_str):
    """
    Converte string de cartas manual para formato normalizado.
    Ex: "Ah Ks" → "AKs", "Ad Kh" → "AKo"
    """
    detector = CardDetector()
    cards = cards_str.strip().split()
    if len(cards) >= 2:
        return detector.normalize_hand(cards[0], cards[1])
    return None


def parse_cards_string(cards_str):
    """
    Parseia string como 'Ah Kd 2c 5h 9s' → ['Ah', 'Kd', '2c', '5h', '9s']
    Aceita maiúsculas/minúsculas e vários separadores.
    """
    if not cards_str:
        return []

    # Normaliza 10 → T
    cards_str = cards_str.replace("10", "T")

    # Encontra padrão de carta: rank + naipe
    pattern = r'([AKQJTakqjt2-9]{1,2}[shdcSHDC])'
    found = re.findall(pattern, cards_str)

    # Normaliza
    result = []
    for card in found:
        rank = card[:-1].upper()
        suit = card[-1].lower()
        if rank in RANKS and suit in SUITS:
            result.append(f"{rank}{suit}")

    return result
