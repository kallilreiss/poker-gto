"""
API routes for Poker Vision GTO.
"""

import io
import traceback

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from api.schemas import AnalysisResult, ErrorResponse
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

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_hand(file: UploadFile = File(...)):
    """
    Receives a poker screenshot and returns a full GTO analysis.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo deve ser uma imagem.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {e}")

    try:
        hero_cards = detect_hole_cards(image)
        board = detect_board(image)
        stacks = detect_stacks(image)
        pot_bb = detect_pot(image)
        num_players = detect_num_players(image)
        position = detect_position(image, num_players)
        street = detect_street(image)
        available_actions = detect_actions(image)

        hero_stack = stacks.get("hero_stack", 0.0)
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

        return AnalysisResult(
            hero_cards=hero_cards,
            board=board,
            position=position,
            street=street,
            pot_bb=pot_bb,
            hero_stack=hero_stack,
            villain_stacks=villain_stacks,
            available_actions=available_actions,
            gto_action=gto.get("action", "CHECK"),
            gto_frequencies=gto.get("frequencies", {}),
            gto_bet_size=gto.get("bet_size"),
            justification=gto.get("justification", ""),
            hand_category=hand_cat,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise: {traceback.format_exc()}")
