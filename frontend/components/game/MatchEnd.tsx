"use client";

import Link from "next/link";

import type { GameSideStats, GameStateDTO } from "@/types/game";

export function MatchEndOverlay({
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
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4 overflow-y-auto">
      <div className="rounded-2xl bg-zinc-900 border border-zinc-700 shadow-2xl px-6 sm:px-10 py-6 sm:py-8 max-w-md w-full text-center space-y-5 my-auto">
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
        {game.stats && <MatchStats p0={game.stats.p0} p1={game.stats.p1} />}
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

function MatchStats({ p0, p1 }: { p0: GameSideStats; p1: GameSideStats }) {
  const rows: [string, (x: GameSideStats) => number][] = [
    ["Trucos pedidos", (x) => x.truco_calls],
    ["Aumentos (6/9/12)", (x) => x.raises],
    ["Apostas aceitas", (x) => x.accepts],
    ["Corridas", (x) => x.runs],
  ];
  return (
    <table className="w-full text-sm tabular-nums text-left">
      <thead className="text-xs text-zinc-400">
        <tr>
          <th className="font-medium py-1" />
          <th className="font-medium py-1 text-right">Você</th>
          <th className="font-medium py-1 text-right">IA</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, f]) => (
          <tr key={label} className="border-t border-zinc-800">
            <td className="py-1 text-zinc-300">{label}</td>
            <td className="py-1 text-right text-zinc-100">{f(p0)}</td>
            <td className="py-1 text-right text-zinc-100">{f(p1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
