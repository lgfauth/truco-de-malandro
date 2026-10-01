"use client";

import Link from "next/link";
import { useEffect } from "react";

import type { PlayersList } from "@/types/runs";

export function OpponentSelect({
  players,
  value,
  onChange,
  disabled,
  className = "",
}: {
  players: PlayersList | null;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      aria-label="Oponente"
      className={`bg-zinc-950/70 border border-emerald-800 rounded-xl px-3 py-2.5 text-base text-zinc-100 w-full ${className}`}
    >
      {players ? (
        <>
          <optgroup label="Níveis">
            {players.levels.map((l) => (
              <option key={l.ref} value={l.ref}>{l.label}</option>
            ))}
          </optgroup>
          {players.models.length > 0 && (
            <optgroup label="Modelos">
              {players.models.map((m) => (
                <option key={m.ref} value={m.ref}>{m.label}</option>
              ))}
            </optgroup>
          )}
          {players.runs.map((r) => (
            <optgroup key={r.run_id} label={`Run ${r.name}`}>
              {r.checkpoints.map((c) => (
                <option key={c.ref} value={c.ref}>{r.name} · {c.kind}</option>
              ))}
            </optgroup>
          ))}
        </>
      ) : (
        <option value="impossivel">Impossível — PPO liga v2</option>
      )}
    </select>
  );
}

export function EventList({ events }: { events: string[] }) {
  if (!events.length) {
    return <p className="text-xs text-zinc-400">Nada aconteceu ainda nesta partida.</p>;
  }
  return (
    <ol className="space-y-1">
      {events.map((e, i) => (
        <li key={`${events.length - i}`} className={`text-xs ${i === 0 ? "text-zinc-100" : "text-zinc-400"}`}>
          {e}
        </li>
      ))}
    </ol>
  );
}

/** Slide-over menu: opponent, new game, rules and the event history. */
export function GameMenu({
  open,
  onClose,
  players,
  opponent,
  onOpponent,
  onNewGame,
  onRules,
  events,
  busy,
}: {
  open: boolean;
  onClose: () => void;
  players: PlayersList | null;
  opponent: string;
  onOpponent: (v: string) => void;
  onNewGame: () => void;
  onRules: () => void;
  events: string[];
  busy: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/50" onClick={onClose}>
      <aside
        className="game-screen w-[min(22rem,88vw)] bg-zinc-900 border-l border-zinc-800 flex flex-col"
        onClick={(e) => e.stopPropagation()}
        aria-label="Menu da partida"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
          <span className="font-semibold">Menu</span>
          <button onClick={onClose} className="text-2xl leading-none text-zinc-400 hover:text-zinc-100 px-2" aria-label="Fechar menu">
            ×
          </button>
        </div>
        <div className="p-4 space-y-3 border-b border-zinc-800">
          <label className="block text-xs text-zinc-400 space-y-1">
            <span>Oponente da próxima partida</span>
            <OpponentSelect players={players} value={opponent} onChange={onOpponent} disabled={busy} />
          </label>
          <button
            onClick={() => {
              onClose();
              onNewGame();
            }}
            disabled={busy}
            className="w-full py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-zinc-900 font-bold"
          >
            Nova partida
          </button>
          <button
            onClick={() => {
              onClose();
              onRules();
            }}
            className="w-full py-2.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 font-semibold"
          >
            📖 Regras
          </button>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto p-4">
          <div className="text-xs uppercase tracking-wide text-zinc-500 mb-2">Histórico</div>
          <EventList events={events} />
        </div>
        <Link href="/" className="px-4 py-3 border-t border-zinc-800 text-sm text-zinc-400 hover:text-zinc-100">
          ← Menu principal
        </Link>
      </aside>
    </div>
  );
}
