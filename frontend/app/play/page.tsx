"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { newGame, nextHand, sendAction } from "@/lib/api";
import { useTrucoSounds } from "@/hooks/useTrucoSounds";
import {
  ACTION_ACCEPT,
  ACTION_CALL_TRUCO,
  ACTION_PLAY_0,
  ACTION_RAISE,
  ACTION_RUN,
  type CardDTO,
  type GameStateDTO,
} from "@/types/game";

const STAKE_LADDER = [1, 3, 6, 9, 12];

function nextStakeAfter(value: number | null | undefined): number | null {
  if (value == null) return null;
  const i = STAKE_LADDER.indexOf(value);
  if (i < 0 || i >= STAKE_LADDER.length - 1) return null;
  return STAKE_LADDER[i + 1];
}

const SUIT_GLYPH = ["♦", "♠", "♥", "♣"];
const RANK_LABEL: Record<number, string> = {
  4: "4", 5: "5", 6: "6", 7: "7",
  10: "Q", 11: "J", 12: "K", 13: "A",
  14: "2", 15: "3",
};

type HandToast = { message: string; tone: "win" | "lose" };

export default function PlayPage() {
  const [game, setGame] = useState<GameStateDTO | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dismissedMatch, setDismissedMatch] = useState<string | null>(null);
  const [handToast, setHandToast] = useState<HandToast | null>(null);
  const [frozenGame, setFrozenGame] = useState<GameStateDTO | null>(null);
  const [events, setEvents] = useState<string[]>([]);

  const sounds = useTrucoSounds();
  const stakeSound: Record<number, () => void> = {
    3: sounds.playTruco,
    6: sounds.playSeis,
    9: sounds.playNove,
    12: sounds.playDoze,
  };
  const prevTerminated = useRef(false);
  const prevGameRef = useRef<GameStateDTO | null>(null);

  // Single diff-driven effect: round sounds, hand-end freeze + toast, and
  // mid-hand event log entries. We compare prev → cur game state once.
  useEffect(() => {
    if (!game) {
      prevGameRef.current = null;
      return;
    }
    const prev = prevGameRef.current;
    prevGameRef.current = game;
    if (!prev) return;
    if (prev.game_id !== game.game_id) {
      setEvents([]);
      return;
    }

    // Round-result sounds and events.
    const prevRounds = prev.rounds ?? [];
    const curRounds = game.rounds ?? [];
    for (let i = 0; i < curRounds.length; i++) {
      const before = prevRounds[i]?.result ?? null;
      const after = curRounds[i]?.result ?? null;
      if (before === null && after !== null) {
        if (after === 0) {
          sounds.playRoundWin();
          setEvents((es) => [`🎯 Rodada ${i + 1}: Você venceu`, ...es]);
        } else if (after === 1) {
          sounds.playRoundLose();
          setEvents((es) => [`🎯 Rodada ${i + 1}: IA venceu`, ...es]);
        } else {
          sounds.playRoundTie();
          setEvents((es) => [`🤝 Rodada ${i + 1}: Empate`, ...es]);
        }
      }
    }

    if (game.terminated) return;

    // Hand just ended — score changed. Detect runner event.
    // Toast + freeze are handled by the hand_ending useEffect below.
    const p0Up = game.scores.p0 > prev.scores.p0;
    const p1Up = game.scores.p1 > prev.scores.p1;
    if (p0Up || p1Up) {
      let runner: number | null = null;
      if (prev.pending_stake != null && prev.truco_caller != null) {
        runner = prev.truco_caller === 0 ? 1 : 0;
      } else if (prev.awaiting_mao11_response && prev.open_hand_for != null) {
        runner = prev.open_hand_for === 0 ? 1 : 0;
      }
      if (runner !== null) {
        const ev = runner === 0 ? "🏃 Você correu" : "🏃 IA correu";
        setEvents((es) => [ev, ...es]);
      }
    }

    // Mid-hand events: truco call and acceptance.
    if ((prev.pending_stake ?? null) == null && (game.pending_stake ?? null) != null) {
      const caller = game.truco_caller;
      if (caller != null) {
        const who = caller === 0 ? "Você" : "IA";
        setEvents((es) => [`🃏 ${who} pediu Truco (vale ${game.pending_stake})`, ...es]);
      }
    }
    if (
      (prev.pending_stake ?? null) != null &&
      (game.pending_stake ?? null) == null &&
      (game.stake ?? 0) > (prev.stake ?? 0)
    ) {
      setEvents((es) => [`✅ Truco aceito — vale ${game.stake} pts`, ...es]);
    }
  }, [game, sounds]);

  useEffect(() => {
    if (game?.terminated && !prevTerminated.current) {
      if (game.match_winner === 0) sounds.playVitoria();
      else sounds.playDerrota();
    }
    prevTerminated.current = game?.terminated ?? false;
  }, [game?.terminated, game?.match_winner, sounds]);

  // hand_ending: backend froze the state after a hand ended. Show a toast,
  // wait 2.5s, then call /game/next-hand to advance and clear the freeze.
  useEffect(() => {
    if (!game?.hand_ending) return;
    const winner = game.hand_winner;
    const p0Won = winner === 0;
    setFrozenGame(game);
    setHandToast({
      message: p0Won ? "Você venceu a mão! 🎉" : "IA venceu a mão.",
      tone: p0Won ? "win" : "lose",
    });
    if (p0Won) sounds.playRoundWin(); else sounds.playRoundLose();
    const id = setTimeout(async () => {
      try {
        const next = await nextHand(game.game_id);
        setGame(next);
      } catch {
        // ignore; game still playable
      } finally {
        setFrozenGame(null);
        setHandToast(null);
      }
    }, 2500);
    return () => clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [game?.hand_ending, game?.game_id]);

  // Lift the freeze after 2.5s.
  useEffect(() => {
    if (!frozenGame || game?.hand_ending) return;
    const id = setTimeout(() => {
      setFrozenGame(null);
      setHandToast(null);
    }, 2500);
    return () => clearTimeout(id);
  }, [frozenGame]);

  async function handleNewGame() {
    setDismissedMatch(null);
    setError(null);
    setBusy(true);
    try {
      setGame(await newGame());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleAction(action: number) {
    if (!game || busy || frozenGame) return;
    // Immediate audio feedback before the round-trip.
    if (action === ACTION_CALL_TRUCO) {
      const next = nextStakeAfter(game.stake ?? 1);
      if (next != null) stakeSound[next]?.();
    } else if (action === ACTION_RAISE) {
      const base = game.pending_stake ?? game.stake ?? 1;
      const next = nextStakeAfter(base);
      if (next != null) stakeSound[next]?.();
    } else if (action === ACTION_ACCEPT) {
      sounds.playAceitar();
    } else if (action === ACTION_RUN) {
      sounds.playCorrer();
    } else if (action >= ACTION_PLAY_0 && action <= ACTION_PLAY_0 + 2) {
      sounds.playCartaJogada();
    }

    setBusy(true);
    setError(null);
    try {
      setGame(await sendAction(game.game_id, action));
    } catch (e) {
      const msg = (e as Error).message;
      // Backend lost the session (process restarted, in-memory GAMES wiped).
      // Reset the local game so the user can start a fresh one.
      if (msg.startsWith("404") || msg.includes("Unknown game_id")) {
        setGame(null);
        setError(
          "Sessão expirada — o backend reiniciou. Clique em Nova partida."
        );
      } else {
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  }

  const legal = new Set(game?.legal_actions ?? []);

  return (
    <main
      className="min-h-screen text-zinc-100 px-3 py-4 sm:px-6 sm:py-8"
      style={{
        backgroundColor: "#1a5c2a",
        backgroundImage: `
          url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.08 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>"),
          radial-gradient(ellipse at center, #1e6b30 0%, #0f3d1a 100%)
        `,
        backgroundAttachment: "fixed",
      }}
    >
      <div className="max-w-5xl mx-auto space-y-6">
        <nav className="flex items-center justify-between">
          <Link href="/" className="text-zinc-300 hover:text-white">
            ← Voltar
          </Link>
          {game && (
            <span className="text-xs text-zinc-300/80">
              Partida {game.game_id.slice(0, 8)}
            </span>
          )}
        </nav>

        <h1 className="text-3xl font-bold">Jogar contra a IA</h1>

        <section className="rounded-xl bg-felt-800/70 border border-emerald-900 p-4">
          <div className="flex flex-wrap gap-2 justify-end">
            <button
              onClick={handleNewGame}
              disabled={busy}
              className="px-4 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-zinc-800 font-semibold"
            >
              Nova partida
            </button>
          </div>
        </section>

        {error && (
          <div className="p-3 rounded bg-rose-950/80 border border-rose-800 text-rose-200 text-sm">
            {error}
          </div>
        )}

        {!game ? (
          <p className="text-zinc-300/80">
            Clique em &quot;Nova partida&quot; para começar.
          </p>
        ) : (
          <Table
            game={frozenGame ?? game}
            legal={frozenGame ? new Set() : legal}
            onAction={handleAction}
            busy={busy || frozenGame !== null}
            handToast={handToast}
            events={events}
          />
        )}
      </div>
      {game?.terminated && dismissedMatch !== game.game_id && (
        <MatchEndOverlay
          game={game}
          busy={busy}
          onNewGame={() => {
            setDismissedMatch(game.game_id);
            handleNewGame();
          }}
        />
      )}
    </main>
  );
}

function MatchEndOverlay({
  game,
  busy,
  onNewGame,
}: {
  game: GameStateDTO;
  busy: boolean;
  onNewGame: () => void;
}) {
  const won = game.match_winner === 0;
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="rounded-2xl bg-zinc-900 border border-zinc-700 shadow-2xl px-10 py-8 max-w-md w-full mx-6 text-center space-y-5">
        <h2
          className={`text-4xl font-extrabold ${
            won ? "text-emerald-400" : "text-rose-400"
          }`}
        >
          {won ? "Você venceu! 🎉" : "A IA venceu."}
        </h2>
        <p className="text-zinc-200 text-lg">
          Placar final:{" "}
          <span className="font-bold">
            {game.scores.p0} × {game.scores.p1}
          </span>
        </p>
        <button
          onClick={onNewGame}
          disabled={busy}
          className="w-full px-5 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-zinc-900 font-bold text-lg"
        >
          Nova partida
        </button>
        <Link
          href="/"
          className="block w-full text-center py-2 text-zinc-400 hover:text-zinc-200 text-sm"
        >
          ← Voltar ao menu
        </Link>
      </div>
    </div>
  );
}

function Table({
  game,
  legal,
  onAction,
  busy,
  handToast,
  events,
}: {
  game: GameStateDTO;
  legal: Set<number>;
  onAction: (a: number) => void;
  busy: boolean;
  handToast: HandToast | null;
  events: string[];
}) {
  const handToastTone =
    handToast?.tone === "win"
      ? "bg-emerald-700/90 border-emerald-400 text-emerald-50"
      : "bg-rose-800/90 border-rose-400 text-rose-50";
  return (
    <div className="relative rounded-2xl border border-emerald-900 bg-felt-800/60 p-6 shadow-inner space-y-6">
      {handToast && (
        <div
          className={`absolute top-4 left-1/2 -translate-x-1/2 z-20 px-4 py-2 rounded-lg text-sm font-bold border shadow-lg ${handToastTone}`}
        >
          {handToast.message}
        </div>
      )}
      <header className="flex flex-wrap items-center gap-2 text-xs">
        <Score label="Você (P0)" value={game.scores.p0} />
        <Score label="IA (P1)" value={game.scores.p1} />
        <Badge label={`Stake: ${game.stake ?? 1}`} />
        {game.pending_stake != null && (
          <Badge tone="amber" label={`Pedido: ${game.pending_stake}`} />
        )}
        {game.iron_hand && <Badge tone="rose" label="Mão de ferro" />}
        {game.awaiting_mao11_response && (
          <Badge tone="amber" label="Mão de 11" />
        )}
        {game.vira && (
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-zinc-300/70">Vira:</span>
            <Card card={game.vira} small />
          </div>
        )}
      </header>

      <section>
        <h3 className="text-xs uppercase tracking-wide text-zinc-300/70 mb-2 flex items-center gap-2">
          Mão da IA
          {game.open_hand_for === 1 && (
            <span className="px-1.5 py-0.5 text-[10px] uppercase rounded bg-amber-700/70 border border-amber-500 text-amber-50">
              Mão de 11
            </span>
          )}
        </h3>
        <div className="flex gap-2">
          {(() => {
            if (game.open_hand_for === 1) {
              return (game.p1_hand ?? []).map((c, i) => (
                <Card key={i} card={c} small />
              ));
            }
            const played = (game.rounds ?? []).reduce(
              (acc, r) =>
                acc + r.plays.filter((p) => p.player === 1).length,
              0
            );
            const remaining = Math.max(0, 3 - played);
            return Array.from({ length: remaining }).map((_, i) => (
              <CardBack key={i} small />
            ));
          })()}
        </div>
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wide text-zinc-300/70 mb-2">
          Rodadas
        </h3>
        <div className="grid grid-cols-3 gap-1">
          {[0, 1, 2].map((i) => {
            const r = game.rounds?.[i];
            return (
              <div
                key={i}
                className="rounded-lg bg-felt-900/70 border border-emerald-950 p-2 min-h-[5rem] overflow-visible"
              >
                <div className="text-xs text-zinc-400 mb-1">Rodada {i + 1}</div>
                <div className="flex flex-row gap-1">
                  {r?.plays.map((p, k) => (
                    <div key={k} className="flex flex-col items-center gap-1">
                      <Card card={p.card} small />
                      <span className="text-[10px] text-zinc-400">
                        {p.player === 0 ? "Você" : "IA"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wide text-zinc-300/70 mb-2">
          Sua mão
        </h3>
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex flex-row gap-2 flex-nowrap flex-1 overflow-x-auto">
            {(game.p0_hand ?? []).map((c, i) => {
              const action = ACTION_PLAY_0 + i;
              const playable = legal.has(action) && !busy;
              return (
                <button
                  key={i}
                  disabled={!playable}
                  onClick={() => onAction(action)}
                  className={`transition-transform ${
                    playable ? "hover:-translate-y-2 cursor-pointer" : "opacity-60"
                  }`}
                >
                  {game.iron_hand ? <CardBack small /> : <Card card={c} small />}
                </button>
              );
            })}
          </div>
          <div className="w-full sm:w-48 flex-shrink-0 h-32 sm:h-40 overflow-y-auto bg-black/20 border border-emerald-900 rounded-lg p-2">
            {events.map((e, i) => (
              <div key={i} className="text-xs text-zinc-300">{e}</div>
            ))}
          </div>
        </div>
      </section>

      {game.awaiting_mao11_response && game.open_hand_for === 1 && (
        <p className="text-sm text-amber-200 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2">
          Mão de 11: o adversário tem 11 pontos e mostrou as cartas. Aceite para jogar valendo 3 pontos ou corra para ceder 1 ponto.
        </p>
      )}

      <section className="flex flex-wrap gap-2">
        {legal.has(ACTION_CALL_TRUCO) && (
          <ActionButton onClick={() => onAction(ACTION_CALL_TRUCO)} disabled={busy}>
            Pedir Truco
          </ActionButton>
        )}
        {legal.has(ACTION_ACCEPT) && (
          <ActionButton
            tone="emerald"
            onClick={() => onAction(ACTION_ACCEPT)}
            disabled={busy}
          >
            Aceitar
          </ActionButton>
        )}
        {legal.has(ACTION_RUN) && (
          <ActionButton
            tone="rose"
            onClick={() => onAction(ACTION_RUN)}
            disabled={busy}
          >
            Correr
          </ActionButton>
        )}
        {legal.has(ACTION_RAISE) && (
          <ActionButton
            tone="amber"
            onClick={() => onAction(ACTION_RAISE)}
            disabled={busy}
          >
            Aumentar
          </ActionButton>
        )}
      </section>


    </div>
  );
}

function Card({ card, small = false }: { card: CardDTO; small?: boolean }) {
  const suit = SUIT_GLYPH[card.suit] ?? "?";
  const isRed = card.suit === 0 || card.suit === 2; // Diamonds, Hearts
  const color = isRed ? "#dc2626" : "#1a1a1a";
  const rank = RANK_LABEL[card.rank] ?? String(card.rank);
  const w = small ? 48 : 70;
  const h = small ? 68 : 100;
  const cornerRank = small ? 11 : 14;
  const centerSuit = small ? "1.4rem" : "2rem";

  return (
    <div
      className="relative rounded-lg bg-white border border-zinc-300 shadow-md select-none"
      style={{ width: w, height: h, color }}
    >
      <div
        className="absolute leading-none flex flex-col items-center"
        style={{ top: 4, left: 6 }}
      >
        <span style={{ fontSize: cornerRank, fontWeight: 700 }}>{rank}</span>
        <span style={{ fontSize: cornerRank, lineHeight: 1 }}>{suit}</span>
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <span style={{ fontSize: centerSuit, lineHeight: 1 }}>{suit}</span>
      </div>
      <div
        className="absolute leading-none flex flex-col items-center"
        style={{ bottom: 4, right: 6, transform: "rotate(180deg)" }}
      >
        <span style={{ fontSize: cornerRank, fontWeight: 700 }}>{rank}</span>
        <span style={{ fontSize: cornerRank, lineHeight: 1 }}>{suit}</span>
      </div>
    </div>
  );
}

function CardBack({ small = false }: { small?: boolean }) {
  const w = small ? 48 : 70;
  const h = small ? 68 : 100;
  return (
    <div
      className="rounded-lg border border-zinc-300 shadow-md select-none"
      style={{
        width: w,
        height: h,
        backgroundColor: "#1e3a8a",
        backgroundImage:
          "repeating-linear-gradient(45deg, rgba(255,255,255,0.14) 0 2px, transparent 2px 9px), repeating-linear-gradient(-45deg, rgba(255,255,255,0.14) 0 2px, transparent 2px 9px)",
      }}
    />
  );
}

function Score({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg bg-felt-900/70 border border-emerald-950 px-3 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-zinc-300/70">
        {label}
      </div>
      <div className="text-lg font-bold">{value}</div>
    </div>
  );
}

function Badge({
  label,
  tone = "zinc",
}: {
  label: string;
  tone?: "zinc" | "amber" | "rose";
}) {
  const cls =
    tone === "amber"
      ? "bg-amber-500/20 border-amber-400 text-amber-200"
      : tone === "rose"
      ? "bg-rose-500/20 border-rose-400 text-rose-200"
      : "bg-zinc-700/40 border-zinc-600 text-zinc-200";
  return (
    <span
      className={`px-2 py-0.5 text-xs rounded border ${cls} font-medium`}
    >
      {label}
    </span>
  );
}

function ActionButton({
  children,
  onClick,
  disabled,
  tone = "zinc",
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  tone?: "zinc" | "emerald" | "rose" | "amber";
}) {
  const cls =
    tone === "emerald"
      ? "bg-emerald-600 hover:bg-emerald-500"
      : tone === "rose"
      ? "bg-rose-700 hover:bg-rose-600"
      : tone === "amber"
      ? "bg-amber-600 hover:bg-amber-500"
      : "bg-zinc-700 hover:bg-zinc-600";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`text-sm px-3 py-1.5 rounded-lg font-semibold disabled:opacity-50 ${cls}`}
    >
      {children}
    </button>
  );
}
