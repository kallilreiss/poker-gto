"""
card_detector.py — PokerStars Detection Engine
Usa HSV color segmentation + contour analysis + template matching
para detectar cartas na tela do PokerStars Mobile.
"""

import cv2
import numpy as np
from PIL import Image
import io
import re

RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["s", "h", "d", "c"]

# ─── PALETA POKERSTARS MOBILE ─────────────────────────────────────────────────
# Cartas têm fundo branco/creme com texto preto(s/c) e vermelho(h/d)
# Fundo da mesa é verde escuro
PS_TABLE_GREEN_LOW  = np.array([35,  40,  30])
PS_TABLE_GREEN_HIGH = np.array([85, 255, 160])

# Fundo da carta: quase branco
PS_CARD_WHITE_LOW  = np.array([0,   0, 160])
PS_CARD_WHITE_HIGH = np.array([180, 90, 255])

# Texto vermelho (hearts / diamonds)
PS_RED_LOW1  = np.array([0,   120, 100])
PS_RED_HIGH1 = np.array([10,  255, 255])
PS_RED_LOW2  = np.array([165, 120, 100])
PS_RED_HIGH2 = np.array([180, 255, 255])

# Texto preto (spades / clubs)
PS_DARK_LOW  = np.array([0,   0,   0])
PS_DARK_HIGH = np.array([180, 80,  90])


class CardDetector:
    def __init__(self):
        self.last_debug = {}

    # ──────────────────────────────────────────────────────────────────────────
    # ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────
    def detect_cards_from_image(self, image_bytes):
        """
        Detecta hole cards e board cards numa screenshot do PokerStars.
        Retorna {"hole_cards": [...], "board_cards": [...], "error": None}
        """
        img = self._load(image_bytes)

        if img is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if img is None:
            return {"hole_cards": [], "board_cards": [], "error": "Imagem inválida"}

        h, w = img.shape[:2]

        # 1. Encontra TODOS os retângulos de cartas na imagem
        all_card_rects = self._find_card_rectangles(img)

        if not all_card_rects:
            return {"hole_cards": [], "board_cards": [],
                    "error": "Nenhuma carta encontrada. Tente com mais luz e sem reflexo."}

        # 2. Separa hole cards (parte inferior) de board cards (centro)
        hole_rects, board_rects = self._split_hole_board(all_card_rects, h)

        # 3. Lê cada carta
        hole_cards  = [self._read_card(img, r) for r in hole_rects[:2]]
        board_cards = [self._read_card(img, r) for r in board_rects[:5]]

        hole_cards  = [c for c in hole_cards  if c]
        board_cards = [c for c in board_cards if c]

        self.last_debug = {
            "all_rects": all_card_rects,
            "hole_rects": hole_rects,
            "board_rects": board_rects,
        }

        return {"hole_cards": hole_cards, "board_cards": board_cards, "error": None}

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — ENCONTRA RETÂNGULOS BRANCOS (CARTAS)
    # ──────────────────────────────────────────────────────────────────────────
    def _find_card_rectangles(self, img):
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Máscara de pixels brancos/claros (fundo da carta)
        white_mask = cv2.inRange(hsv, PS_CARD_WHITE_LOW, PS_CARD_WHITE_HIGH)

        # Fecha buracos dentro das cartas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area   = cw * ch
            aspect = cw / ch if ch > 0 else 0

            # Filtros: tamanho mínimo e proporção de carta
            min_area = (w * h) * 0.0005
            max_area = (w * h) * 0.03

            if min_area < area < max_area and 0.45 < aspect < 1.0:
                # Verifica que pelo menos 60% do retângulo é branco
                roi_mask = white_mask[y:y+ch, x:x+cw]
                fill = cv2.countNonZero(roi_mask) / area
                if fill > 0.55:
                    rects.append((x, y, cw, ch))

        # Remove duplicatas muito próximas
        rects = self._deduplicate_rects(rects)
        # Ordena da esquerda para direita
        return sorted(rects, key=lambda r: r[0])

    def _deduplicate_rects(self, rects, overlap_thresh=0.5):
        """Remove retângulos muito sobrepostos."""
        if not rects:
            return rects
        keep = []
        for r in rects:
            dominated = False
            for k in keep:
                if self._iou(r, k) > overlap_thresh:
                    dominated = True
                    break
            if not dominated:
                keep.append(r)
        return keep

    def _iou(self, a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1+aw, ay1+ah
        bx2, by2 = bx1+bw, by1+bh
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = aw*ah + bw*bh - inter
        return inter/union if union > 0 else 0

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — SEPARA HOLE CARDS E BOARD
    # ──────────────────────────────────────────────────────────────────────────
    def _split_hole_board(self, rects, img_height):
        """
        Ajustado para fotos do PokerStars Desktop tiradas pelo celular.
        """
        hole = []
        board = []

        for r in rects:
            x, y, w, h = r
            center_y = y + h / 2

            if center_y > img_height * 0.58:
                hole.append(r)
            elif img_height * 0.18 < center_y < img_height * 0.55:
                board.append(r)

        hole = sorted(hole, key=lambda r: r[0])
        board = sorted(board, key=lambda r: r[0])

        if len(hole) > 2:
            hole = hole[:2]

        if len(board) > 5:
            board = board[:5]

        return hole, board

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — LÊ RANK E NAIPE DE UMA CARTA
    # ──────────────────────────────────────────────────────────────────────────
    def _read_card(self, img, rect):
        x, y, cw, ch = rect
        # Padding pequeno para não cortar bordas
        pad = max(2, int(min(cw, ch) * 0.04))
        x  = max(0, x - pad)
        y  = max(0, y - pad)
        x2 = min(img.shape[1], x + cw + pad*2)
        y2 = min(img.shape[0], y + ch + pad*2)

        card_img = img[y:y2, x:x2]
        if card_img.size == 0:
            return None

        h, w = card_img.shape[:2]

        # Canto superior esquerdo: rank fica em ~25% x 35%
        rank_roi = card_img[0:int(h*0.38), 0:int(w*0.48)]
        # Logo abaixo: naipe em ~35-65% y, 0-45% x
        suit_roi = card_img[int(h*0.35):int(h*0.68), 0:int(w*0.48)]

        rank = self._detect_rank(rank_roi, card_img)
        suit = self._detect_suit_color(suit_roi, card_img)

        if rank and suit:
            return f"{rank}{suit}"
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # RANK DETECTION — análise de pixels do glifo
    # ──────────────────────────────────────────────────────────────────────────
    def _detect_rank(self, roi, full_card):
        if roi is None or roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Inverte: texto escuro → branco no threshold
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

        # Remove ruído
        k = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filtra contornos pequenos demais
        h_roi, w_roi = roi.shape[:2]
        min_c_area = (h_roi * w_roi) * 0.01
        valid = [c for c in contours if cv2.contourArea(c) > min_c_area]
        if not valid:
            return None

        # Bounding box de todos os glifos juntos
        all_x, all_y, all_w, all_h = [], [], [], []
        for c in valid:
            x, y, cw, ch = cv2.boundingRect(c)
            all_x.append(x); all_y.append(y)
            all_w.append(x+cw); all_h.append(y+ch)

        gx = min(all_x); gy = min(all_y)
        gx2 = max(all_w); gy2 = max(all_h)
        glyph_w = gx2 - gx
        glyph_h = gy2 - gy

        if glyph_h == 0:
            return None

        glyph_aspect = glyph_w / glyph_h
        n_contours   = len(valid)
        total_area   = sum(cv2.contourArea(c) for c in valid)
        fill_ratio   = total_area / (glyph_w * glyph_h) if glyph_w * glyph_h > 0 else 0

        # Extrai glifo para análise de forma
        glyph_img = thresh[gy:gy2, gx:gx2]
        return self._classify_rank(glyph_img, glyph_aspect, n_contours,
                                    fill_ratio, glyph_w, glyph_h)

    def _classify_rank(self, glyph, aspect, n_contours, fill, gw, gh):
        """
        Classificação baseada em características geométricas dos glifos
        de fontes típicas de poker (Arial/sans-serif bold).
        """
        if glyph is None or glyph.size == 0:
            return None

        # Projeta pixels horizontalmente e verticalmente
        h_proj = np.sum(glyph, axis=1).astype(float)  # por linha
        v_proj = np.sum(glyph, axis=0).astype(float)  # por coluna

        h_proj /= (h_proj.max() + 1e-9)
        v_proj /= (v_proj.max() + 1e-9)

        # Número de cruzamentos (cruza 50% de intensidade)
        def crossings(proj, thresh=0.3):
            above = proj > thresh
            return int(np.sum(np.diff(above.astype(int)) != 0))

        h_cross = crossings(h_proj)
        v_cross = crossings(v_proj)

        # Divide em terços verticais
        t = len(h_proj) // 3
        top_density    = float(np.mean(h_proj[:t]))
        middle_density = float(np.mean(h_proj[t:2*t]))
        bottom_density = float(np.mean(h_proj[2*t:]))

        # Divide em terços horizontais
        t2 = len(v_proj) // 3
        left_density   = float(np.mean(v_proj[:t2]))
        center_density = float(np.mean(v_proj[t2:2*t2]))
        right_density  = float(np.mean(v_proj[2*t2:]))

        # ── Regras de classificação ────────────────────────────────────────

        # "10" — dois contornos (1 e 0), aspecto largo
        if n_contours >= 2 and aspect > 1.2:
            return "T"

        # "A" — aspecto triangular, denso no topo e base, buraco no meio
        if (top_density > 0.3 and bottom_density > 0.3
                and middle_density < top_density * 0.7
                and aspect < 1.1 and n_contours <= 2):
            return "A"

        # "K" — muitos cruzamentos horizontais, assimétrico
        if h_cross >= 6 and right_density > left_density * 1.2:
            return "K"

        # "Q" — oval com abertura, fill alto
        if fill > 0.65 and aspect > 0.75 and h_cross <= 4 and n_contours <= 2:
            if bottom_density > top_density * 0.8:
                return "Q"

        # "J" — fino, pouco largo, curvado em baixo
        if aspect < 0.65 and bottom_density > middle_density:
            return "J"

        # "8" — dois buracos (2 contornos internos), simétrico
        if n_contours >= 2 and 0.55 < aspect < 0.95 and abs(top_density - bottom_density) < 0.15:
            return "8"

        # "9" — buraco no topo, aberto em baixo
        if top_density > bottom_density * 1.3 and fill > 0.5:
            return "9"

        # "6" — buraco em baixo, aberto no topo
        if bottom_density > top_density * 1.3 and fill > 0.5:
            return "6"

        # "4" — cruzamento no meio, assimétrico
        if middle_density > top_density and middle_density > bottom_density and aspect < 1.0:
            return "4"

        # "5" — parte superior recta, inferior curva
        if top_density > 0.6 and bottom_density > 0.5 and middle_density < 0.5:
            return "5"

        # "3" — dois arcos à direita
        if right_density > left_density * 1.3 and h_cross >= 4:
            return "3"

        # "2" — arco em cima, linha em baixo
        if top_density > 0.5 and bottom_density > 0.7 and middle_density < 0.4:
            return "2"

        # "7" — linha horizontal no topo, diagonal
        if top_density > 0.7 and bottom_density < top_density * 0.5:
            return "7"

        # "T" (10) — aspecto mais largo
        if aspect >= 0.9 and n_contours == 1:
            return "T"

        # Fallback por aspecto
        if aspect > 1.0:   return "T"
        if aspect > 0.85:  return "K"
        if aspect > 0.70:  return "Q"
        return "J"

    # ──────────────────────────────────────────────────────────────────────────
    # SUIT DETECTION — cor do símbolo do naipe
    # ──────────────────────────────────────────────────────────────────────────
    def _detect_suit_color(self, suit_roi, full_card):
        if suit_roi is None or suit_roi.size == 0:
            return self._detect_suit_from_full_card(full_card)

        hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, PS_RED_LOW1, PS_RED_HIGH1)
        red_mask2 = cv2.inRange(hsv, PS_RED_LOW2, PS_RED_HIGH2)
        red_mask  = cv2.bitwise_or(red_mask1, red_mask2)
        dark_mask = cv2.inRange(hsv, PS_DARK_LOW, PS_DARK_HIGH)

        red_px  = cv2.countNonZero(red_mask)
        dark_px = cv2.countNonZero(dark_mask)

        # Se a cor dominante for vermelha → hearts ou diamonds
        if red_px > dark_px and red_px > 3:
            return self._distinguish_red_suit(suit_roi)

        # Se for escura → spades ou clubs
        if dark_px > 3:
            return self._distinguish_dark_suit(suit_roi)

        # Fallback: analisa carta inteira
        return self._detect_suit_from_full_card(full_card)

    def _detect_suit_from_full_card(self, card_img):
        """Analisa a carta inteira para detectar cor."""
        hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, PS_RED_LOW1, PS_RED_HIGH1)
        red2 = cv2.inRange(hsv, PS_RED_LOW2, PS_RED_HIGH2)
        red  = cv2.countNonZero(cv2.bitwise_or(red1, red2))
        dark = cv2.countNonZero(cv2.inRange(hsv, PS_DARK_LOW, PS_DARK_HIGH))
        if red > dark:
            return "h"
        return "s"

    def _distinguish_red_suit(self, roi):
        """
        Hearts vs Diamonds:
        - Hearts: forma de coração — mais largo no centro, pontiagudo em baixo
        - Diamonds: losango — mais simétrico, pontiagudo em cima e em baixo
        """
        if roi.size == 0:
            return "h"
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "h"

        cnt = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0:
            return "h"

        aspect = cw / ch

        # Diamonds são mais quadrados/verticais; hearts mais largos
        if aspect > 1.05:
            return "h"   # Hearts são mais largos que altos
        return "d"       # Diamonds são mais altos que largos

    def _distinguish_dark_suit(self, roi):
        """
        Spades vs Clubs:
        - Clubs: 3 bolinhas + haste — fill disperso
        - Spades: formato de pá — mais compacto
        """
        if roi.size == 0:
            return "s"
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Clubs geralmente têm 2+ contornos distintos (as 3 bolinhas)
        if len(contours) >= 2:
            return "c"
        return "s"

    # ──────────────────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────────────────
    def _load(self, image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except:
            return None

    def normalize_hand(self, card1, card2):
        """Normaliza 2 cartas para formato GTO: AKs, QJo, TTo..."""
        if not card1 or not card2 or len(card1) < 2 or len(card2) < 2:
            return None

        r1, s1 = card1[:-1].upper(), card1[-1].lower()
        r2, s2 = card2[:-1].upper(), card2[-1].lower()
        r1 = r1.replace("10", "T")
        r2 = r2.replace("10", "T")

        RANK_ORDER = {r: i for i, r in enumerate(RANKS)}
        if r1 not in RANK_ORDER or r2 not in RANK_ORDER:
            return None

        # Maior rank primeiro
        if RANK_ORDER[r1] > RANK_ORDER[r2]:
            r1, r2, s1, s2 = r2, r1, s2, s1

        if r1 == r2:
            return f"{r1}{r2}o"
        elif s1 == s2:
            return f"{r1}{r2}s"
        else:
            return f"{r1}{r2}o"

    def extract_game_info(self, image_bytes):
        return self.detect_cards_from_image(image_bytes)


def parse_cards_string(cards_str):
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
