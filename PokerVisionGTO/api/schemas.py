"""
Pydantic schemas for the Poker Vision GTO API.
"""

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    hero_cards: list[str] = Field(default_factory=list, description="Hero's hole cards")
    board: list[str] = Field(default_factory=list, description="Community cards")
    position: str = Field(default="Unknown", description="Hero's table position")
    street: str = Field(default="Preflop", description="Current street")
    pot_bb: float = Field(default=0.0, description="Pot size in BB")
    hero_stack: float = Field(default=0.0, description="Hero's stack in BB")
    villain_stacks: list[float] = Field(default_factory=list, description="Villain stacks in BB")
    available_actions: list[str] = Field(default_factory=list, description="Available actions")
    gto_action: str = Field(default="CHECK", description="GTO recommended action")
    gto_frequencies: dict[str, int] = Field(default_factory=dict, description="Action frequencies %")
    gto_bet_size: str | None = Field(default=None, description="Recommended bet size")
    justification: str = Field(default="", description="Strategic justification")
    hand_category: str = Field(default="", description="Hand strength category")


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
