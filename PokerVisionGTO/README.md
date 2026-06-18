# ♠ Poker Vision GTO

Analise screenshots de mesas de poker e receba recomendações GTO automáticas usando Computer Vision e OCR.

---

## Funcionalidades

- **Detecção de cartas** — Reconhece as cartas do Hero e do board via OpenCV + EasyOCR
- **Detecção de stacks** — Lê os tamanhos de pilha de todos os jogadores em BB
- **Detecção de pote** — Extrai o valor do pote automaticamente
- **Detecção de posição** — Identifica a posição do Hero (BTN, CO, HJ, MP, UTG, SB, BB) usando o dealer button
- **Detecção de street** — Preflop / Flop / Turn / River pela quantidade de cartas no board
- **Motor GTO** — Consulta tabelas GTO pré-carregadas e retorna ação recomendada com frequências
- **Interface Streamlit** — Upload, preview, análise e histórico exportável em CSV/JSON

---

## Estrutura do Projeto

```
PokerVisionGTO/
├── card_detector/
│   ├── detect_hole_cards.py   # Cartas do Hero
│   ├── detect_board.py        # Cartas da mesa
│   ├── rank_detector.py       # Detecção de rank via OCR
│   └── suit_detector.py       # Detecção de naipe via cor
├── table_detector/
│   ├── detect_stacks.py       # Stacks dos jogadores
│   ├── detect_pot.py          # Tamanho do pote
│   ├── detect_positions.py    # Posição do Hero
│   ├── detect_dealer_button.py# Localização do dealer button
│   ├── detect_actions.py      # Botões de ação disponíveis
│   ├── detect_players.py      # Número de jogadores
│   └── detect_street.py       # Street atual
├── ocr/
│   ├── stack_reader.py        # Utilitário OCR para stacks
│   ├── pot_reader.py          # Utilitário OCR para pote
│   └── name_reader.py         # Utilitário OCR para nomes
├── gto_engine/
│   ├── lookup.py              # Motor de consulta GTO
│   └── strategy_engine.py     # Avaliador de mão + tabelas GTO
├── api/
│   ├── main.py                # FastAPI app
│   ├── routes.py              # Endpoint /analyze
│   └── schemas.py             # Pydantic schemas
├── web/
│   └── streamlit_app.py       # Interface principal
├── tests/
│   └── test_all.py            # Suite de testes (pytest)
├── requirements.txt
├── run.sh
└── README.md
```

---

## Instalação

### Pré-requisitos

- Python 3.12+
- pip

### Clone e instale

```bash
git clone https://github.com/seu-usuario/PokerVisionGTO.git
cd PokerVisionGTO
pip install -r requirements.txt
```

> **Nota:** O EasyOCR fará o download dos modelos de linguagem automaticamente na primeira execução (~500 MB).

---

## Execução Local

### Opção 1 — Script único (API + Streamlit)

```bash
chmod +x run.sh
./run.sh
```

- Streamlit: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

### Opção 2 — Separado

```bash
# Terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
streamlit run web/streamlit_app.py
```

---

## API

### POST `/api/v1/analyze`

Recebe uma imagem e retorna análise completa.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@screenshot.png"
```

**Resposta:**

```json
{
  "hero_cards": ["7c", "4c"],
  "board": ["4h", "Js", "Kd"],
  "position": "BB",
  "street": "Flop",
  "pot_bb": 5.67,
  "hero_stack": 37.0,
  "villain_stacks": [74.7, 163.1, 99.0, 164.4, 454.2],
  "available_actions": ["CHECK", "BET"],
  "gto_action": "CHECK",
  "gto_frequencies": {"CHECK": 82, "BET": 18},
  "gto_bet_size": "33%",
  "justification": "Mão fraca OOP. Check predominante.",
  "hand_category": "One Pair"
}
```

---

## Testes

```bash
pytest tests/ -v --tb=short
```

Para cobertura:

```bash
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Deploy — Streamlit Cloud

1. Faça fork do repositório no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte o repositório
4. Defina `web/streamlit_app.py` como Main file path
5. Clique em **Deploy**

---

## Mesas Suportadas

| Formato | Jogadores |
|---------|-----------|
| 6-max   | 2–6       |
| 9-max   | 2–9       |

---

## Plataformas Testadas

- PokerStars (tema padrão)
- GGPoker
- 888poker

> **Atenção:** Esta ferramenta é destinada exclusivamente ao estudo e análise de mãos fora de sessões ativas. Verifique os Termos de Serviço da sua plataforma antes de utilizar durante o jogo.

---

## Tecnologias

| Camada          | Tecnologia         |
|-----------------|--------------------|
| Backend         | FastAPI + Uvicorn  |
| Frontend        | Streamlit          |
| Computer Vision | OpenCV + NumPy     |
| OCR             | EasyOCR            |
| Imagens         | Pillow             |
| Dados           | Pandas             |
| Testes          | pytest             |

---

## Licença

MIT © 2024
