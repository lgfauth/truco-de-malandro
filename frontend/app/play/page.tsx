"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { GameMenu, EventList, OpponentSelect } from "@/components/game/GameMenu";
import { MatchEndOverlay } from "@/components/game/MatchEnd";
import { RulesModal } from "@/components/game/RulesModal";
import { GameTable, RoundsHistory, nextStakeAfter, type HandToast } from "@/components/game/Table";
import { useTrucoSounds } from "@/hooks/useTrucoSounds";
import { continueGame, getArenaPlayers, newGame, nextHand, sendAction } from "@/lib/api";
import {
  ACTION_ACCEPT,
  ACTION_CALL_TRUCO,
  ACTION_PLAY_0,
  ACTION_RAISE,
  ACTION_RUN,
  type GameStateDTO,
} from "@/types/game";
import type { PlayersList } from "@/types/runs";

const MAX_EVENTS = 60;
// How long the finished round stays on the table before the AI opens the next.
const AI_PAUSE_MS = 3000;

export default function PlayPage() {
  const [game, setGame] = useState<GameStateDTO | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dismissedMatch, setDismissedMatch] = useState<string | null>(null);
  const [handToast, setHandToast] = useState<HandToast | null>(null);
  const [frozenGame, setFrozenGame] = useState<GameStateDTO | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [rulesOpen, setRulesOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [opponent, setOpponent] = useState("impossivel");
  const [players, setPlayers] = useState<PlayersList | null>(null);

  useEffect(() => {
    getArenaPlayers()
      .then((p) => {
        setPlayers(p);
        setOpponent((cur) => cur || p.default);
      })
      .catch(() => setPlayers(null)); // selector falls back to the default level
  }, []);

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
          setEvents((es) => [`🎯 Rodada ${i + 1}: Você venceu`, ...es].slice(0, MAX_EVENTS));
        } else if (after === 1) {
          sounds.playRoundLose();
          setEvents((es) => [`🎯 Rodada ${i + 1}: IA venceu`, ...es].slice(0, MAX_EVENTS));
        } else {
          sounds.playRoundTie();
          setEvents((es) => [`🤝 Rodada ${i + 1}: Empate`, ...es].slice(0, MAX_EVENTS));
        }
      }
    }

    if (game.terminated) return;

    // Hand just ended — score changed. Detect runner event.
    // Toast + freeze are handled by the hand_ending useEffect below.
    const p0Up = game.scores.p0 > prev.scores.p0;
    const p1Up = game.scores.p1 > prev.scores.p1;
    if (p0Up || p1Up) {
      // Prefer the frozen-hand fields on the just-ended hand (game when
      // hand_ending=true) — they reflect the *latest* truco_caller after
      // any raise chain. prev's snapshot can be stale when a raise was
      // processed server-side in the same round-trip.
      const trucoCaller = game.hand_ending && game.truco_caller != null
        ? game.truco_caller
        : prev.truco_caller;
      const mao11Player = game.hand_ending && game.mao11_player != null
        ? game.mao11_player
        : prev.mao11_player;

      let runner: number | null = null;
      if ((prev.pending_stake != null || game.hand_ending) && trucoCaller != null) {
        // Runner = adversary of whoever called the truco most recently.
        runner = trucoCaller === 0 ? 1 : 0;
      } else if (prev.awaiting_mao11_response && mao11Player != null) {
        // Mão de 11: only the player at 11 can run.
        runner = mao11Player;
      }
      if (runner !== null) {
        const ev = runner === 0 ? "🏃 Você correu" : "🏃 IA correu";
        setEvents((es) => [ev, ...es].slice(0, MAX_EVENTS));
      }
    }

    // Mid-hand events: truco call and acceptance.
    if ((prev.pending_stake ?? null) == null && (game.pending_stake ?? null) != null) {
      const caller = game.truco_caller;
      if (caller != null) {
        const who = caller === 0 ? "Você" : "IA";
        setEvents((es) => [`🃏 ${who} pediu Truco (vale ${game.pending_stake})`, ...es].slice(0, MAX_EVENTS));
      }
    }
    if (
      (prev.pending_stake ?? null) != null &&
      (game.pending_stake ?? null) == null &&
      (game.stake ?? 0) > (prev.stake ?? 0)
    ) {
      setEvents((es) => [`✅ Truco aceito — vale ${game.stake} pts`, ...es].slice(0, MAX_EVENTS));
    }
  }, [game, sounds]);

  useEffect(() => {
    if (game?.terminated && !prevTerminated.current) {
      if (game.match_winner === 0) sounds.playVitoria();
      else sounds.playDerrota();
    }
    prevTerminated.current = game?.terminated ?? false;
  }, [game?.terminated, game?.match_winner, sounds]);

  // ai_pending: the AI won (or tied while leading) a round and opens the
  // next one. Keep the finished round on the table, then let the AI move.
  useEffect(() => {
    if (!game?.ai_pending) return;
    const id = setTimeout(async () => {
      try {
        setGame(await continueGame(game.game_id));
      } catch (e) {
        setError((e as Error).message);
      }
    }, AI_PAUSE_MS);
    return () => clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [game?.ai_pending, game?.game_id, game?.rounds?.length]);

  // hand_ending: backend froze the state after a hand ended. Show a toast,
  // wait 2.5s, then call /game/next-hand to advance and clear the freeze.
  useEffect(() => {
    if (!game?.hand_ending) return;
    const winner = game.hand_winner;
    const p0Won = winner === 0;
    setFrozenGame(game);
    if (game.hand_drawn) {
      setHandToast({ message: "Mão empatada — ninguém pontua.", tone: "draw" });
      sounds.playRoundTie();
    } else {
      setHandToast({
        message: p0Won ? "Você venceu a mão! 🎉" : "IA venceu a mão.",
        tone: p0Won ? "win" : "lose",
      });
      if (p0Won) sounds.playRoundWin(); else sounds.playRoundLose();
    }
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
      setGame(await newGame(opponent));
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

  const shown = frozenGame ?? game;
  const closeMenu = useCallback(() => setMenuOpen(false), []);

  return (
    <main className="game-screen felt-bg text-zinc-100 flex flex-col overflow-hidden">
      <div className="flex-1 min-h-0 w-full max-w-6xl mx-auto flex gap-4 px-2 sm:px-4 lg:py-3">
        <div className="flex-1 min-w-0 min-h-0 flex flex-col">
          {error && (
            <div role="alert" className="shrink-0 mt-2 p-2 rounded-lg bg-rose-950/90 border border-rose-800 text-rose-200 text-sm">
              {error}
            </div>
          )}
          {!shown ? (
            <StartScreen
              players={players}
              opponent={opponent}
              onOpponent={setOpponent}
              onStart={handleNewGame}
              onRules={() => setRulesOpen(true)}
              busy={busy}
            />
          ) : (
            <GameTable
              game={shown}
              legal={frozenGame ? EMPTY : legal}
              onAction={handleAction}
              busy={busy || frozenGame !== null}
              handToast={handToast}
              lastEvent={events[0]}
              onMenu={() => setMenuOpen(true)}
            />
          )}
        </div>
        {shown && (
          <aside className="hidden lg:flex w-72 shrink-0 flex-col gap-3 min-h-0 py-2">
            <div className="rounded-2xl bg-black/25 p-4 space-y-2">
              <div className="text-xs uppercase tracking-wide text-emerald-200/70">Oponente</div>
              <div className="font-semibold">{shown.opponent?.label ?? "—"}</div>
              <button
                onClick={handleNewGame}
                disabled={busy}
                className="w-full mt-1 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-zinc-900 font-bold text-sm"
              >
                Nova partida
              </button>
            </div>
            <div className="rounded-2xl bg-black/25 p-4">
              <div className="text-xs uppercase tracking-wide text-emerald-200/70 mb-2">Rodadas desta mão</div>
              <RoundsHistory game={shown} />
            </div>
            <div className="rounded-2xl bg-black/25 p-4 flex-1 min-h-0 overflow-y-auto">
              <div className="text-xs uppercase tracking-wide text-emerald-200/70 mb-2">Histórico</div>
              <EventList events={events} />
            </div>
          </aside>
        )}
      </div>

      <GameMenu
        open={menuOpen}
        onClose={closeMenu}
        players={players}
        opponent={opponent}
        onOpponent={setOpponent}
        onNewGame={handleNewGame}
        onRules={() => setRulesOpen(true)}
        events={events}
        busy={busy}
      />
      {rulesOpen && <RulesModal onClose={() => setRulesOpen(false)} />}
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

const EMPTY = new Set<number>();

function StartScreen({
  players,
  opponent,
  onOpponent,
  onStart,
  onRules,
  busy,
}: {
  players: PlayersList | null;
  opponent: string;
  onOpponent: (v: string) => void;
  onStart: () => void;
  onRules: () => void;
  busy: boolean;
}) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-6 px-4 text-center">
      <div>
        <h1 className="text-3xl sm:text-4xl font-extrabold">Truco de Malandro</h1>
        <p className="text-emerald-100/80 mt-2">Truco Paulista, você contra a IA. Primeiro a 12 vence.</p>
      </div>
      <div className="w-full max-w-sm space-y-3">
        <label className="block text-left text-sm text-emerald-100/80 space-y-1">
          <span>Oponente</span>
          <OpponentSelect players={players} value={opponent} onChange={onOpponent} disabled={busy} />
        </label>
        <button
          onClick={onStart}
          disabled={busy}
          className="w-full py-3.5 rounded-2xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-zinc-900 font-extrabold text-lg shadow-lg"
        >
          {busy ? "Embaralhando…" : "Nova partida"}
        </button>
        <div className="flex gap-3">
          <button onClick={onRules} className="flex-1 py-2.5 rounded-xl bg-black/30 hover:bg-black/40 font-semibold">
            📖 Regras
          </button>
          <Link href="/" className="flex-1 py-2.5 rounded-xl bg-black/30 hover:bg-black/40 font-semibold">
            Menu
          </Link>
        </div>
      </div>
    </div>
  );
}
