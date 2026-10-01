"use client";

import { useCallback, useEffect, useState } from "react";

import { JobProgress } from "@/components/arena";
import { Button, Card, Empty, ErrorBox, Loading, Page, StatusBadge } from "@/components/ui";
import { createArenaJob, getArenaJob, getArenaPlayers, listArenaJobs } from "@/lib/api";
import { ciText, dateTime, pct, playerLabel } from "@/lib/format";
import type { ArenaJob, MatchupSummary, PlayersList } from "@/types/runs";

const GAME_OPTIONS = [100, 500, 1000, 2000];

function PlayerSelect({
  label,
  value,
  onChange,
  players,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  players: PlayersList;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-400 min-w-0 flex-1">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 w-full"
      >
        <optgroup label="Básicos">
          {players.builtin.map((b) => (
            <option key={b} value={b}>{playerLabel(b)}</option>
          ))}
        </optgroup>
        <optgroup label="Modelos publicados (models/)">
          {players.models.map((m) => (
            <option key={m.ref} value={m.ref}>{m.label}</option>
          ))}
        </optgroup>
        {players.runs.map((r) => (
          <optgroup key={r.run_id} label={`Run ${r.name} (${r.run_id})`}>
            {r.checkpoints.map((c) => (
              <option key={c.ref} value={c.ref}>{r.name} · {c.kind}</option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

export default function ArenaPage() {
  const [players, setPlayers] = useState<PlayersList | null>(null);
  const [jobs, setJobs] = useState<ArenaJob[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [a, setA] = useState("");
  const [b, setB] = useState("rule");
  const [games, setGames] = useState(500);
  const [seed, setSeed] = useState(123);
  const [job, setJob] = useState<ArenaJob | null>(null);

  const loadJobs = useCallback(() => {
    listArenaJobs(20).then(setJobs).catch((e) => setError((e as Error).message));
  }, []);

  const load = useCallback(() => {
    setError(null);
    getArenaPlayers()
      .then((p) => {
        setPlayers(p);
        setA((cur) => cur || p.models.find((m) => m.ref.includes("truco_liga_v2"))?.ref || p.builtin[0]);
      })
      .catch((e) => setError((e as Error).message));
    loadJobs();
  }, [loadJobs]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    const t = setInterval(async () => {
      try {
        const j = await getArenaJob(job.id);
        setJob(j);
        if (j.status === "done" || j.status === "failed") loadJobs();
      } catch (e) {
        setError((e as Error).message);
      }
    }, 800);
    return () => clearInterval(t);
  }, [job, loadJobs]);

  async function run() {
    setError(null);
    try {
      setJob(await createArenaJob({ a, b, games, seed }));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const busy = job != null && (job.status === "queued" || job.status === "running");

  return (
    <Page
      title="Arena"
      subtitle="Dois jogadores, mãos fixas de avaliação (separadas do treino), cada mão jogada dos dois lados."
    >
      {error && <ErrorBox error={error} onRetry={load} />}
      {!players && !error && <Loading />}

      {players && (
        <Card title="Novo confronto">
          <div className="flex flex-col sm:flex-row gap-3 sm:items-end">
            <PlayerSelect label="Jogador A" value={a} onChange={setA} players={players} />
            <span className="text-zinc-500 text-sm self-center hidden sm:block pb-2">vs</span>
            <PlayerSelect label="Jogador B" value={b} onChange={setB} players={players} />
          </div>
          <div className="flex flex-wrap items-end gap-3 mt-3">
            <label className="flex flex-col gap-1 text-xs text-zinc-400">
              Partidas
              <select
                value={games}
                onChange={(e) => setGames(Number(e.target.value))}
                className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100"
              >
                {GAME_OPTIONS.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-zinc-400">
              Semente das mãos
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value) || 0)}
                className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 w-28"
              />
            </label>
            <Button tone="emerald" onClick={run} disabled={busy || !a || !b}>
              {busy ? "Jogando…" : "Rodar"}
            </Button>
            <span className="text-xs text-zinc-500">
              IC 95% com {games} partidas: ±{Math.round(98 / Math.sqrt(games))} pontos no pior caso.
            </span>
          </div>
          {job && (
            <div className="mt-4">
              <JobProgress job={job} />
            </div>
          )}
        </Card>
      )}

      {job?.status === "done" && job.result && <ResultCard summary={job.result} />}

      <Card title="Confrontos recentes">
        {!jobs && <Loading />}
        {jobs && jobs.length === 0 && <Empty>Nenhum confronto rodado pela interface ainda.</Empty>}
        {jobs && jobs.length > 0 && (
          <div className="overflow-x-auto -mx-4 px-4">
            <table className="w-full text-sm min-w-[560px]">
              <thead className="text-xs text-zinc-400 text-left">
                <tr>
                  <th className="py-2 pr-3 font-medium">Quando</th>
                  <th className="py-2 pr-3 font-medium">A</th>
                  <th className="py-2 pr-3 font-medium">B</th>
                  <th className="py-2 pr-3 font-medium text-right">Partidas</th>
                  <th className="py-2 pr-3 font-medium text-right">Vitórias de A</th>
                  <th className="py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr
                    key={j.id}
                    className="border-t border-zinc-800 hover:bg-zinc-800/50 cursor-pointer"
                    onClick={() => setJob(j)}
                  >
                    <td className="py-2 pr-3 text-xs text-zinc-400">{dateTime(j.created_at)}</td>
                    <td className="py-2 pr-3">{playerLabel(j.a)}</td>
                    <td className="py-2 pr-3">{playerLabel(j.b)}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{j.games}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      {j.result ? (
                        <>
                          {pct(j.result.win_rate_a)}{" "}
                          <span className="text-xs text-zinc-500">{ciText(j.result.ci95)}</span>
                        </>
                      ) : "—"}
                    </td>
                    <td className="py-2"><StatusBadge status={j.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </Page>
  );
}

function ResultCard({ summary: s }: { summary: MatchupSummary }) {
  const [lo, hi] = s.ci95;
  const rows: [string, (x: MatchupSummary["a_stats"]) => string][] = [
    ["Pontos por mão", (x) => x.points_per_hand.toFixed(2)],
    ["Trucos por mão", (x) => x.truco_rate.toFixed(3)],
    ["Corrida por aposta enfrentada", (x) => pct(x.run_rate)],
    ["Mãos vencidas", (x) => String(x.hands_won)],
    ["Ações ilegais", (x) => String(x.illegal)],
    ["Fallbacks", (x) => String(x.fallbacks)],
    ["Erros", (x) => String(x.errors)],
  ];
  return (
    <Card title="Resultado" subtitle={`${s.games} partidas · ${s.hands} mãos (${s.drawn_hands} empatadas) · ${s.elapsed_s}s`}>
      <div className="grid sm:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] gap-6 items-start">
        <div>
          <div className="text-xs uppercase tracking-wide text-zinc-500">Vitórias de A</div>
          <div className="text-4xl font-bold text-zinc-100 mt-1">{pct(s.win_rate_a)}</div>
          <div className="text-sm text-zinc-400">
            {s.a_wins} de {s.games} · {ciText(s.ci95)}
          </div>
          {/* CI on a 0–100% track with the 50% mark */}
          <div className="relative h-3 mt-3 rounded bg-zinc-800" aria-hidden>
            <div className="absolute top-0 bottom-0 w-px bg-zinc-500" style={{ left: "50%" }} />
            <div
              className="absolute top-0.5 bottom-0.5 rounded bg-[#3987e5]/60"
              style={{ left: `${lo * 100}%`, width: `${Math.max(0.5, (hi - lo) * 100)}%` }}
            />
            <div className="absolute -top-0.5 h-4 w-0.5 bg-white" style={{ left: `${s.win_rate_a * 100}%` }} />
          </div>
          <div className="flex justify-between text-[10px] text-zinc-500 mt-1">
            <span>0%</span><span>50%</span><span>100%</span>
          </div>
          <div className="text-xs text-zinc-400 mt-3 space-y-0.5">
            <div>A como P0 (abre a 1ª mão): {pct(s.by_seat.a_as_p0?.win_rate)}</div>
            <div>A como P1: {pct(s.by_seat.a_as_p1?.win_rate)}</div>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabular-nums">
            <thead className="text-xs text-zinc-400">
              <tr>
                <th className="text-left font-medium py-1" />
                <th className="text-right font-medium py-1 px-2 max-w-[10rem] truncate">A · {playerLabel(s.a)}</th>
                <th className="text-right font-medium py-1 pl-2 max-w-[10rem] truncate">B · {playerLabel(s.b)}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, f]) => (
                <tr key={label} className="border-t border-zinc-800">
                  <td className="py-1.5 text-zinc-300">{label}</td>
                  <td className="py-1.5 px-2 text-right">{f(s.a_stats)}</td>
                  <td className="py-1.5 pl-2 text-right">{f(s.b_stats)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  );
}
