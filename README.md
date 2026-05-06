# Truco de Malandro

Agente de aprendizado por reforço que joga **Truco Paulista** *heads-up* (1 vs 1). O agente é treinado com **MaskablePPO** (ação mascarada para garantir que apenas ações legais sejam exploradas) e disponibilizado via uma API REST + WebSocket. O frontend permite assistir o treino em tempo real e jogar contra a IA.

---

## O que o projeto entrega

| Funcionalidade | Descrição |
|---|---|
| Motor de regras completo | Truco Paulista com baralho de 40 cartas, manilhas, mão de 11, mão de ferro e escadinha de apostas 1→3→6→9→12 |
| Ambiente Gymnasium | Espaço de observação (24 floats), espaço de ação Discrete (7) com máscara dinâmica, recompensa esparsa ±1.0 ao fim da partida |
| Treino com MaskablePPO | Treinamento contra oponente aleatório ou *self-play*; checkpoints automáticos; métricas (win rate, recompensa média) transmitidas por WebSocket |
| API REST | FastAPI com endpoints de treino, jogo e health check |
| Frontend Next.js | Dashboard de métricas (gráficos em tempo real) e tabuleiro interativo (humano vs IA) com efeitos sonoros sintetizados |
| Deploy no Railway | Dois serviços independentes (backend Python + frontend Node) com configuração pronta |

---

## Estrutura do projeto

```
truco-de-malandro/
├── backend/
│   ├── main.py               # Ponto de entrada FastAPI (CORS, rotas, health check)
│   ├── requirements.txt      # Dependências Python
    ├── train_all.py          # Script para treinar o modelo truco_ppo_1M (1M steps)
│   ├── debug_env.py          # Script manual para inspecionar o ambiente (5 episódios)
│   ├── Procfile              # Comando de start para o Railway
│   ├── railway.toml          # Configuração de health check do Railway
│   ├── agent/
│   │   └── trainer.py        # Lógica MaskablePPO, self-play, callback de avaliação
│   ├── api/
│   │   └── routes.py         # Endpoints REST (/train/*, /game/*, /health) e WS /ws/metrics
    ├── models/               # Checkpoint truco_ppo_1M.zip gerado pelo treino
│   └── truco/
│       ├── game.py           # Motor de regras puro (TrucoGame, dataclasses, ações)
│       └── env.py            # Wrapper Gymnasium (TrucoEnv, observação, recompensa, máscara)
└── frontend/
    ├── package.json          # React 18, Next.js 15, Recharts, TypeScript 5
    ├── next.config.mjs
    ├── tailwind.config.ts    # Tema com verde-feltro customizado
    ├── app/
    │   ├── layout.tsx        # Layout raiz, tema escuro, locale pt-BR
    │   ├── page.tsx          # Home com links Watch / Play
    │   ├── play/page.tsx     # Tabuleiro interativo (humano vs IA)
    │   └── watch/page.tsx    # Dashboard de métricas do treino
    ├── lib/
    │   └── api.ts            # Cliente HTTP/WS centralizado
    ├── types/
    │   └── game.ts           # Interfaces TypeScript (GameStateDTO, EvalMetric, etc.)
    └── hooks/
        └── useTrucoSounds.ts # Sintetizador de efeitos sonoros via Web Audio API
```

---

## Como o código funciona

### Motor de regras (`truco/game.py`)

`TrucoGame` encapsula toda a lógica do Truco Paulista:

- **Baralho**: 40 cartas (ranks 4–3 + reis/damas/etc. × 4 naipes). A força das cartas segue a hierarquia do Truco Paulista e as manilhas são calculadas dinamicamente a partir da **vira**.
- **Manilha**: rank seguinte à vira em ordem cíclica; em empate entre manilhas, desempate por naipe (Paus > Copas > Espadas > Ouros).
- **Apostas (stake)**: escadinha 1 → 3 → 6 → 9 → 12. Um jogador pode pedir truco, aceitar, correr ou aumentar.
- **Mão de 11**: quando um jogador chega a 11 pontos, o adversário vê as cartas dele e decide aceitar (joga por 3) ou correr (cede 1 ponto) antes de qualquer carta ser jogada.
- **Mão de ferro**: ambos com 11 pontos — vale 1 ponto, sem truco e com cartas escondidas.
- `legal_actions()` retorna apenas as `PlayerAction`s válidas para o turno atual.
- `step(action)` aplica a ação e atualiza o estado (rodadas, stake, placar, vencedor).

### Ambiente Gymnasium (`truco/env.py`)

`TrucoEnv` adapta o `TrucoGame` para o ciclo `reset/step` do Gymnasium:

- **Observação**: vetor de 24 floats em [0, 1] — cartas da mão (P0 e P1), vira, jogadas das 3 rodadas, placar normalizado, flags de estado (mão de 11, mão de ferro, stake pendente etc.).
- **Máscara de ação**: vetor binário de tamanho 7 calculado dinamicamente a cada passo; o MaskablePPO nunca amostra ações ilegais.
- **Recompensa**: esparsa — +1.0 ao P0 ganhar a partida, −1.0 ao perder, 0 nos demais passos.
- **Self-play** (`_p1_view()`): inverte a perspectiva para que o oponente congelado jogue como P1.

### Treino (`agent/trainer.py`)

- **Algoritmo**: `sb3_contrib.MaskablePPO` com política MLP.
- **Hiperparâmetros principais**: `lr=3e-4`, `n_steps=512`, `batch_size=32`, `ent_coef=0.05`.
- **`TrainingCallback`**: avalia a cada 50 episódios contra um oponente aleatório (100 jogos), armazena métricas na lista `metrics` (win_rate, mean_reward, mean_steps).
- **`_ControlCallback`**: checa o `stop_flag` para honrar pausas/resets via API.
- **Self-play**: o oponente congelado é atualizado a cada 200 episódios com os pesos atuais do agente.
- **Smoke test** (`smoke_test()`): 2 000 timesteps + validação de que a entropia é não-zero antes do treino completo.

### API (`api/routes.py` + `main.py`)

FastAPI exposta em `http://localhost:8000`:

#### Treino

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/train/start` | Inicia treino em thread daemon; aceita `{ total_timesteps, save_path }` |
| `POST` | `/train/pause` | Para o treino graciosamente (timeout 5 s) |
| `POST` | `/train/reset` | Encerra thread e limpa artefatos |
| `GET` | `/train/status` | Retorna `{ running, paused, timestep, episode, latest_metric }` |
| `WS` | `/ws/metrics` | Transmite métricas a cada 2 s enquanto o treino roda |

#### Jogo

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/game/new` | Cria partida com o agente padrão (1M steps, cheat mode) |
| `POST` | `/game/action` | Executa ação do P0 e deixa a IA jogar seus turnos; aceita `{ game_id, action: 0–6 }` |
| `GET` | `/game/state` | Retorna estado atual; aceita `?game_id=<str>` |
| `POST` | `/game/next-hand` | Avança para a próxima mão após `hand_ending=true` |
| `GET` | `/health` | Health check (`{ status: "ok" }`) |

#### Ações (action IDs)

| ID | Nome | Condição |
|----|------|----------|
| 0–2 | Jogar carta 0/1/2 | Qualquer turno de jogo |
| 3 | Pedir Truco | Sem stake pendente |
| 4 | Aceitar | Stake pendente ou mão de 11 |
| 5 | Correr | Stake pendente ou mão de 11 |
| 6 | Aumentar | Stake pendente (contra-aumento) |

### Frontend

- **`/`** — Home com links para Watch e Play.
- **`/watch`** — Conecta ao WebSocket `/ws/metrics`, exibe gráficos de *win rate* e *recompensa média* (Recharts) e botões para iniciar/pausar/resetar o treino.
- **`/play`** — Tabuleiro completo: cartas do P0 (clicáveis), cartas do P1 (escondidas ou reveladas na mão de 11), placar, rodadas, log de eventos, botões de ação e overlay de fim de partida. Efeitos sonoros sintetizados via Web Audio API para cada evento do jogo.

---

## Jogar online

Acesse diretamente pelo link do Railway, sem precisar rodar nada localmente:

**[https://truco-malandro.up.railway.app](https://truco-malandro.up.railway.app)**

---

## Pré-requisitos

### Backend
- Python **3.10+**
- pip

### Frontend
- Node.js **18+**
- npm **9+**

---

## Instalação e execução local

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

API disponível em `http://localhost:8000`.

### 2. Treinar o modelo

O checkpoint fica em `backend/models/truco_ppo_1M.zip`. Para gerá-lo:

```bash
cd backend
python train_all.py
```

Isso treina o modelo com **1 000 000 de timesteps** e salva em `models/truco_ppo_1M.zip`.

### 3. Frontend

Crie o arquivo `frontend/.env.local` apontando para o backend local:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Em seguida:

```bash
cd frontend
npm install
npm run dev
```

App disponível em `http://localhost:3000`.

---

## Deploy no Railway

O Railway hospeda os dois serviços de forma independente a partir do mesmo repositório, cada um com seu `railway.toml` e `Procfile`. O Nixpacks detecta Python pelo `requirements.txt` e Node pelo `package.json`.

### 1. Backend

1. **New Project → Deploy from GitHub repo** → selecione o repositório → **Root Directory = `backend`**.
2. A variável `PORT` é **injetada automaticamente** pelo Railway; o `Procfile` já a utiliza.
3. O health check está configurado em `/health` via `railway.toml`.
4. Aguarde o build e anote a URL pública gerada (ex.: `https://truco-backend.up.railway.app`).

> **Atenção**: os checkpoints em `backend/models/` precisam estar commitados no repositório ou gerados em um build step, pois o Railway não persiste volumes por padrão no plano gratuito. Para treinos longos prefira gerar os `.zip` localmente e commitá-los.

### 2. Frontend

1. **New Service** no mesmo projeto → mesmo repositório → **Root Directory = `frontend`**.
2. Adicione a variável de ambiente na **UI do Railway**:
   - `NEXT_PUBLIC_API_URL` = URL pública do backend (ex.: `https://truco-backend.up.railway.app`)
3. O health check está configurado em `/` via `railway.toml`.
4. Após o build, abra a URL do frontend e jogue.

### Observações

- Variáveis `NEXT_PUBLIC_*` são embutidas no build do Next.js. Se a URL do backend mudar, é necessário um **redeploy do frontend**.
- O backend armazena partidas **em memória** (dicionário Python). Reiniciar o serviço descarta partidas em andamento.
- O treino roda em uma **thread daemon** dentro do processo do servidor. Para treinos longos, prefira um plano com mais memória ou rode o treino localmente e suba apenas os checkpoints.
