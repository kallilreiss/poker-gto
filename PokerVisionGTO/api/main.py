"""
Poker Vision GTO — FastAPI backend entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Poker Vision GTO",
    description="Analise screenshots de poker e receba recomendações GTO.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"status": "ok", "service": "Poker Vision GTO"}


@app.get("/health")
def health():
    return {"status": "healthy"}
