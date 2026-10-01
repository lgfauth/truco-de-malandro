"use client";

import { memo, useEffect, useState } from "react";

import {
  ACTION_ACCEPT,
  ACTION_CALL_TRUCO,
  ACTION_PLAY_0,
  ACTION_RAISE,
  ACTION_RUN,
  type GameStateDTO,
  type RoundDTO,
} from "@/types/game";

import { CardBack, CardSlot, PlayingCard, RANK_LABEL, manilhaRank } from "./Card";

export type HandToast = { message: string; tone: "win" | "lose" | "draw" };

const STAKE_LADDER = [1, 3, 6, 9, 12];

export function nextStakeAfter(value: number | null | undefined): number | null {
  if (value == null) return null;
  const i = STAKE_LADDER.indexOf(value);
  if (i < 0 || i >= STAKE_LADDER.length - 1) return null;
  return STAKE_LADDER[i + 1];
}

// --- Top bar --------------------------------------------------------------------
function TopBar({ game, onMenu }: { game: GameStateDTO; onMenu: () => void }) {
  return (
    <header className="shrink-0 flex items-center gap-2 sm:gap-3 px-1 py-1.5">
      <button
        onClick={onMenu}
        className="h-10 w-10 grid place-items-center rounded-xl bg-black/25 hover:bg-black/40 text-xl"
        aria-label="Abrir menu"
      >
        ≡
      </button>
      <div className="flex items-center gap-2 rounded-xl bg-black/25 px-3 py-1" aria-label="Placar">
        <span className="text-[11px] uppercase text-emerald-200/80">Você</span>
        <span className="text-2xl font-extrabold tabular-nums">{game.scores.p0}</span>
        <span className="text-zinc-400">×</span>
        <span className="text-2xl font-extrabold tabular-nums">{game.scores.p1}</span>
        <span className="text-[11px] uppercase text-emerald-200/80">IA</span>
      </div>
      <div className="ml-auto flex flex-wrap justify-end items-center gap-1.5 text-xs font-semibold">
        {game.iron_hand && <Chip tone="rose">Mão de ferro</Chip>}
        {game.mao11_player != null && (
          <Chip tone="amber">Mão de 11{game.mao11_player === 1 ? " · IA" : ""}</Chip>
        )}
        {game.pending_stake != null ? (
          <Chip tone="amber">Pedido: {game.pending_stake}</Chip>
        ) : (
          <Chip>Vale {game.stake ?? 1}</Chip>
        )}
      </div>
    </header>
  );
}

function Chip({ children, tone = "zinc" }: { children: React.ReactNode; tone?: "zinc" | "amber" | "rose" }) {
  const cls =
    tone === "amber"
      ? "bg-amber-400 text-zinc-900"
      : tone === "rose"
      ? "bg-rose-600 text-white"
      : "bg-black/30 text-zinc-100";
  return <span className={`px-2.5 py-1 rounded-full whitespace-nowrap ${cls}`}>{children}</span>;
}

// --- Opponent -------------------------------------------------------------------
function OpponentZone({ game }: { game: GameStateDTO }) {
  const left = game.p1_cards_left ?? 3;
  return (
    <section className="shrink-0 flex items-center justify-center gap-3 py-1" aria-label="Mão da IA">
      <div className="flex">
        {Array.from({ length: left }).map((_, i) => (
          <CardBack key={i} size="back" className={i ? "-ml-[3%]" : ""} />
        ))}
        {left === 0 && <div className="pcard pcard-back" />}
      </div>
      <div className="text-xs text-emerald-100/80 leading-tight">
        <div className="font-semibold text-zinc-100">IA</div>
        <div className="max-w-[9rem] truncate">{game.opponent?.label ?? ""}</div>
      </div>
    </section>
  );
}

// --- Center: the round in focus ---------------------------------------------------
const RESULT_TEXT = ["você venceu", "IA venceu", "empate"];

function RoundMarkers({
  rounds,
  focus,
  onPick,
}: {
  rounds: RoundDTO[];
  focus: number;
  onPick: (i: number) => void;
}) {
  return (
    <div className="flex gap-1.5" role="group" aria-label="Rodadas">
      {[0, 1, 2].map((i) => {
        const r = rounds[i];
        const res = r?.result;
        const label = res === 0 ? "✓" : res === 1 ? "✗" : res === 2 ? "=" : r ? "•" : "";
        const tone =
          res === 0
            ? "bg-emerald-500 text-white"
            : res === 1
            ? "bg-rose-600 text-white"
            : res === 2
            ? "bg-zinc-500 text-white"
            : r
            ? "bg-black/30 text-amber-300"
            : "bg-black/15 text-white/30";
        return (
          <button
            key={i}
            disabled={!r || r.plays.length === 0}
            onClick={() => onPick(i)}
            className={`h-7 min-w-[3.25rem] px-2 rounded-full text-xs font-bold flex items-center justify-center gap-1 ${tone} ${
              focus === i && r ? "ring-2 ring-white/80" : ""
            }`}
            aria-label={`Rodada ${i + 1}${res != null ? `: ${RESULT_TEXT[res]}` : ""}`}
          >
            {i + 1}ª {label}
          </button>
        );
      })}
    </div>
  );
}

function Center({
  game,
  handToast,
  myTurnToPlay,
}: {
  game: GameStateDTO;
  handToast: HandToast | null;
  myTurnToPlay: boolean;
}) {
  const rounds = game.rounds ?? [];
  const [picked, setPicked] = useState<number | null>(null);

  // A picked past round is shown for a moment, then the live round returns.
  useEffect(() => {
    if (picked == null) return;
    const t = setTimeout(() => setPicked(null), 3500);
    return () => clearTimeout(t);
  }, [picked]);
  useEffect(() => setPicked(null), [game.game_id, rounds.length]);

  // Live focus: the current round, or the one that just finished while the
  // next has no cards yet (so the result stays visible).
  let focus = rounds.length - 1;
  if (focus > 0 && rounds[focus].plays.length === 0) focus -= 1;
  if (picked != null && rounds[picked]) focus = picked;
  const round = rounds[focus];
  const ai = round?.plays.find((p) => p.player === 1)?.card;
  const me = round?.plays.find((p) => p.player === 0)?.card;
  const result = round?.result ?? null;
  const manilha = manilhaRank(game.vira);
  const toastTone =
    handToast?.tone === "win"
      ? "bg-emerald-600 text-white"
      : handToast?.tone === "draw"
      ? "bg-zinc-600 text-white"
      : "bg-rose-600 text-white";

  return (
    <section className="relative flex-1 min-h-0 flex flex-col items-center justify-center gap-2" aria-label="Mesa">
      <RoundMarkers rounds={rounds} focus={focus} onPick={(i) => setPicked(i === picked ? null : i)} />
      <div className="flex items-end justify-center gap-[3vw] sm:gap-6">
        {game.vira && (
          <figure className="flex flex-col items-center gap-1 mr-[2vw] sm:mr-4">
            <PlayingCard card={game.vira} size="vira" className="-rotate-6" />
            <figcaption className="text-[11px] text-emerald-100/80">
              vira · manilha {manilha != null ? RANK_LABEL[manilha] : "?"}
            </figcaption>
          </figure>
        )}
        {[
          { who: "IA", card: ai, lost: result === 0 },
          { who: "Você", card: me, lost: result === 1 },
        ].map(({ who, card, lost }) => (
          <figure key={who} className="flex flex-col items-center gap-1">
            {card ? (
              <PlayingCard
                key={`${focus}-${card.label}`}
                card={card}
                size="table"
                manilha={card.rank === manilha}
                className={`card-in ${lost ? "opacity-55 saturate-50" : ""}`}
              />
            ) : (
              <CardSlot size="table" label={who === "Você" && myTurnToPlay ? "sua vez" : ""} />
            )}
            <figcaption className="text-xs font-semibold text-emerald-50/90">{who}</figcaption>
          </figure>
        ))}
      </div>
      <div className="h-5 text-xs text-emerald-50/90" aria-live="polite">
        {result != null && `Rodada ${focus + 1}: ${RESULT_TEXT[result]}`}
      </div>
      {handToast && (
        <div className="absolute inset-0 grid place-items-center pointer-events-none">
          <div className={`px-5 py-3 rounded-2xl text-lg font-extrabold shadow-2xl card-in ${toastTone}`} role="status">
            {handToast.message}
          </div>
        </div>
      )}
    </section>
  );
}

// --- Actions ----------------------------------------------------------------------
function ActionBar({
  game,
  legal,
  busy,
  onAction,
}: {
  game: GameStateDTO;
  legal: Set<number>;
  busy: boolean;
  onAction: (a: number) => void;
}) {
  const buttons: { a: number; label: string; cls: string }[] = [];
  const raiseTo = nextStakeAfter(game.pending_stake);
  const trucoTo = nextStakeAfter(game.stake ?? 1);
  const mao11 = game.awaiting_mao11_response && game.mao11_player === 0;
  if (legal.has(ACTION_ACCEPT))
    buttons.push({ a: ACTION_ACCEPT, label: mao11 ? "Jogar (vale 3)" : "Aceitar", cls: "bg-emerald-600 hover:bg-emerald-500" });
  if (legal.has(ACTION_RAISE))
    buttons.push({ a: ACTION_RAISE, label: raiseTo ? `Pedir ${raiseTo}` : "Aumentar", cls: "bg-amber-500 hover:bg-amber-400 text-zinc-900" });
  if (legal.has(ACTION_RUN))
    buttons.push({ a: ACTION_RUN, label: "Correr", cls: "bg-rose-600 hover:bg-rose-500" });
  if (legal.has(ACTION_CALL_TRUCO))
    buttons.push({ a: ACTION_CALL_TRUCO, label: trucoTo && trucoTo > 3 ? `Pedir ${trucoTo}` : "Truco!", cls: "bg-amber-500 hover:bg-amber-400 text-zinc-900" });

  return (
    <section className="shrink-0 min-h-[3.25rem] flex items-center justify-center gap-2 px-1" aria-label="Ações">
      {buttons.map((b) => (
        <button
          key={b.a}
          onClick={() => onAction(b.a)}
          disabled={busy}
          className={`flex-1 max-w-[11rem] h-12 rounded-xl font-extrabold text-base shadow-lg active:scale-95 transition-transform disabled:opacity-50 ${b.cls}`}
        >
          {b.label}
        </button>
      ))}
    </section>
  );
}

// --- Hand -------------------------------------------------------------------------
function Hand({
  game,
  legal,
  busy,
  onAction,
}: {
  game: GameStateDTO;
  legal: Set<number>;
  busy: boolean;
  onAction: (a: number) => void;
}) {
  const manilha = manilhaRank(game.vira);
  return (
    <section className="shrink-0 flex justify-center gap-[2.5vw] sm:gap-4 pt-1 pb-2" aria-label="Sua mão">
      {(game.p0_hand ?? []).map((c, i) => {
        const action = ACTION_PLAY_0 + i;
        const playable = legal.has(action) && !busy;
        return (
          <button
            key={`${c.label}-${i}`}
            disabled={!playable}
            onClick={() => onAction(action)}
            aria-label={`Jogar ${c.label}`}
            className={`rounded-[9%/6.5%] transition-transform duration-150 ${
              playable
                ? "[@media(hover:hover)]:hover:-translate-y-3 active:scale-95 shadow-[0_0_0_3px_#fde047,0_0_18px_rgba(253,224,71,0.45)] cursor-pointer"
                : ""
            }`}
          >
            {game.iron_hand ? (
              <CardBack size="hand" />
            ) : (
              <PlayingCard card={c} size="hand" manilha={c.rank === manilha} />
            )}
          </button>
        );
      })}
      {(game.p0_hand ?? []).length === 0 && <div className="pcard pcard-hand" />}
    </section>
  );
}

// --- Status line ------------------------------------------------------------------
function StatusLine({
  game,
  lastEvent,
  legal,
  busy,
}: {
  game: GameStateDTO;
  lastEvent: string | undefined;
  legal: Set<number>;
  busy: boolean;
}) {
  let text = lastEvent ?? "";
  let tone = "text-emerald-50/90";
  if (game.awaiting_mao11_response && game.mao11_player === 0) {
    text = "Mão de 11: olhe suas cartas — jogar vale 3; correr dá 1 ponto à IA.";
    tone = "text-amber-200 font-semibold";
  } else if (game.ai_pending) {
    const closed = (game.rounds ?? []).filter((r) => r.result != null);
    const res = closed[closed.length - 1]?.result;
    text =
      res === 1
        ? `IA venceu a ${closed.length}ª rodada — ela abre a próxima…`
        : `Rodada ${closed.length} empatada — a IA abre a próxima…`;
  } else if (!game.hand_ending && !game.terminated) {
    if (legal.has(ACTION_ACCEPT) && game.pending_stake != null) {
      text = `A IA pediu ${game.pending_stake}! Aceita, corre ou aumenta?`;
      tone = "text-amber-200 font-semibold";
    } else if (legal.has(ACTION_PLAY_0) && !busy) {
      text = lastEvent ? `${lastEvent} · sua vez` : "Sua vez — toque numa carta";
    } else if (busy) {
      text = "IA pensando…";
    }
  }
  return (
    <div className={`shrink-0 h-6 px-2 text-center text-xs sm:text-sm truncate ${tone}`} aria-live="polite">
      {text}
    </div>
  );
}

function GameTableImpl({
  game,
  legal,
  busy,
  onAction,
  handToast,
  lastEvent,
  onMenu,
}: {
  game: GameStateDTO;
  legal: Set<number>;
  busy: boolean;
  onAction: (a: number) => void;
  handToast: HandToast | null;
  lastEvent: string | undefined;
  onMenu: () => void;
}) {
  const myTurnToPlay = legal.has(ACTION_PLAY_0) || legal.has(ACTION_PLAY_0 + 1) || legal.has(ACTION_PLAY_0 + 2);
  return (
    <div className="flex flex-col h-full min-h-0">
      <TopBar game={game} onMenu={onMenu} />
      <OpponentZone game={game} />
      <Center game={game} handToast={handToast} myTurnToPlay={myTurnToPlay && !busy} />
      <StatusLine game={game} lastEvent={lastEvent} legal={legal} busy={busy} />
      <ActionBar game={game} legal={legal} busy={busy} onAction={onAction} />
      <Hand game={game} legal={legal} busy={busy} onAction={onAction} />
    </div>
  );
}

export const GameTable = memo(GameTableImpl);

/** Desktop sidebar: every round of the current hand with its cards. */
export function RoundsHistory({ game }: { game: GameStateDTO }) {
  const manilha = manilhaRank(game.vira);
  const rounds = (game.rounds ?? []).filter((r) => r.plays.length);
  if (!rounds.length) return <p className="text-xs text-zinc-400">Nenhuma carta jogada nesta mão.</p>;
  return (
    <ol className="space-y-2">
      {rounds.map((r, i) => (
        <li key={i} className="flex items-center gap-2">
          <span className="text-xs text-zinc-400 w-7">{i + 1}ª</span>
          {r.plays.map((p, k) => (
            <span key={k} className="flex flex-col items-center">
              <PlayingCard card={p.card} size="mini" manilha={p.card.rank === manilha} />
              <span className="text-[10px] text-zinc-400">{p.player === 0 ? "você" : "IA"}</span>
            </span>
          ))}
          <span className="text-xs text-zinc-300 ml-1">{r.result != null ? RESULT_TEXT[r.result] : ""}</span>
        </li>
      ))}
    </ol>
  );
}
