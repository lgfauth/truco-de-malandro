"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  MetricChart,
  SERIES,
  WinRateScale,
  opponentColor,
  winRateColor,
  type Row,
} from "@/components/charts";
import {
  Button,
  Card,
  Empty,
  ErrorBox,
  Loading,
  Page,
  Stat,
  StatusBadge,
  Warning,
} from "@/components/ui";
import {
  createArenaJob,
  getArenaJob,
  getArenaPlayers,
  getRun,
  getRunCheckpoints,
  getRunMatrix,
  getRunMetrics,
} from "@/lib/api";
import { JobProgress } from "@/components/arena";
import { fitElo, type PairResult } from "@/lib/elo";
import { ciText, dateTime, duration, pct, playerLabel, steps } from "@/lib/format";
import type {
  ArenaJob,
  CheckpointInfo,
  Matrix,
  PlayersList,
  RunMetric,
  RunSummary,
} from "@/types/runs";

// Train and eval win rates against the same opponent further apart than this
// (over the last evaluations) suggest the policy is fitting the training mix.
const GAP_WARN = 0.15;
const GAP_WINDOW = 3;

export default function RunPage() {
  const params = useParams<{ id: string }>();
  const id = decodeURIComponent(params.id);
  const [run, setRun] = useState<RunSummary | null>(null);
  const [metrics, setMetrics] = useState<RunMetric[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [r, m] = await Promise.all([getRun(id), getRunMetrics(id)]);
      setRun(r.run);
      setMetrics(m.items);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  // Live refresh while the run is going.
  useEffect(() => {
    if (run?.status !== "running") return;
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [run?.status, load]);

  if (error && !run) {
    return (
      <Page title="Run">
        <ErrorBox error={error} onRetry={load} />
      </Page>
    );
  }
  if (!run || !metrics) {
    return (
      <Page title="Run">
        <Loading />
      </Page>
    );
  }

  return (
    <Page
      title={run.name}
      subtitle={
        <span className="flex flex-wrap items-center gap-2">
          <StatusBadge status={run.status} />
          <span>{run.id}</span>
          <span>· início {dateTime(run.started_at)}</span>
          {run.git?.commit && (
            <span>
              · commit {run.git.commit.slice(0, 7)}
              {run.git.dirty ? " (com alterações locais)" : ""}
            </span>
          )}
        </span>
      }
      actions={<Link href="/runs" className="text-sm text-zinc-400 hover:text-zinc-100">← todas as runs</Link>}
    >
      {error && <ErrorBox error={error} onRetry={load} />}
      {run.error && <ErrorBox error={`A run falhou: ${run.error}`} />}

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat label="Passos" value={steps(run.timesteps)} hint={`de ${steps(run.total_timesteps)}`} />
        <Stat label="Duração" value={duration(run.duration_s)} />
        <Stat
          label="Melhor checkpoint"
          value={run.best ? pct(run.best.score) : "—"}
          hint={run.best ? `em ${steps(run.best.timestep)} · vs conjunto fixo` : undefined}
        />
        <Stat
          label="Observação / seed"
          value={`${run.config.obs_version ?? run.obs_version} · ${run.config.seed ?? "—"}`}
          hint={run.config.eval_games ? `${run.config.eval_games} partidas por avaliação` : undefined}
        />
      </section>

      <ConfigDetails run={run} />

      {metrics.length === 0 ? (
        <Empty>Nenhuma avaliação ainda. A primeira sai no primeiro intervalo de avaliação.</Empty>
      ) : (
        <>
          <DivergenceWarning metrics={metrics} />
          <WinRateByOpponent metrics={metrics} />
          <TrainVsEval metrics={metrics} />
          <BehaviourCharts metrics={metrics} />
          <Strength metrics={metrics} />
        </>
      )}
      <MatrixSection runId={id} />
    </Page>
  );
}

function ConfigDetails({ run }: { run: RunSummary }) {
  const hp = run.config.hyperparams ?? {};
  return (
    <details className="rounded-xl border border-zinc-800 bg-zinc-900 p-4 text-sm">
      <summary className="cursor-pointer text-zinc-200 font-semibold">Configuração</summary>
      <div className="grid sm:grid-cols-2 gap-4 mt-3 text-xs">
        <div>
          {run.config.init_from && (
            <>
              <div className="text-zinc-400 mb-1">Continuação de</div>
              <div className="mb-3">{playerLabel(run.config.init_from)}</div>
            </>
          )}
          <div className="text-zinc-400 mb-1">Liga (oponentes no treino)</div>
          {Object.entries(run.config.league ?? {}).map(([k, v]) => (
            <div key={k} className="flex justify-between border-t border-zinc-800 py-1">
              <span>{playerLabel(k)}</span>
              <span className="tabular-nums">{Math.round(v * 100)}%</span>
            </div>
          ))}
          <div className="text-zinc-400 mt-3 mb-1">Oponentes da avaliação</div>
          <div>{(run.config.eval_opponents ?? []).map(playerLabel).join(" · ")}</div>
        </div>
        <div>
          <div className="text-zinc-400 mb-1">Hiperparâmetros</div>
          {Object.entries(hp).map(([k, v]) => (
            <div key={k} className="flex justify-between gap-4 border-t border-zinc-800 py-1">
              <span className="text-zinc-300">{k}</span>
              <span className="tabular-nums text-zinc-100">{JSON.stringify(v)}</span>
            </div>
          ))}
        </div>
      </div>
    </details>
  );
}

function evalKeys(metrics: RunMetric[]): string[] {
  const keys: string[] = [];
  for (const m of metrics) for (const k of Object.keys(m.eval)) if (!keys.includes(k)) keys.push(k);
  return keys;
}

function DivergenceWarning({ metrics }: { metrics: RunMetric[] }) {
  const recent = metrics.slice(-GAP_WINDOW);
  const gaps: { opp: string; train: number; evalWr: number }[] = [];
  for (const opp of ["random", "rule"]) {
    const pairs = recent
      .map((m) => ({ t: m.train.by_opponent[opp], e: m.eval[opp] }))
      .filter((p) => p.t && p.t.win_rate != null && p.t.games >= 20 && p.e);
    if (pairs.length < Math.min(GAP_WINDOW, metrics.length)) continue;
    const train = pairs.reduce((s, p) => s + (p.t.win_rate as number), 0) / pairs.length;
    const evalWr = pairs.reduce((s, p) => s + p.e.win_rate, 0) / pairs.length;
    if (train - evalWr > GAP_WARN) gaps.push({ opp, train, evalWr });
  }
  if (!gaps.length) return null;
  return (
    <Warning title="Treino e avaliação divergindo">
      {gaps.map((g) => (
        <div key={g.opp}>
          Contra {playerLabel(g.opp)}: {pct(g.train)} no treino × {pct(g.evalWr)} na avaliação
          (média das últimas {GAP_WINDOW} avaliações). O agente vence bem mais nas partidas de
          treino do que em mãos novas — sinal de sobreajuste à mistura de treino.
        </div>
      ))}
    </Warning>
  );
}

function WinRateByOpponent({ metrics }: { metrics: RunMetric[] }) {
  const keys = evalKeys(metrics);
  return (
    <Card
      title="Taxa de vitória por oponente"
      subtitle="Avaliação em mãos fixas separadas do treino, cada mão jogada dos dois lados. Banda = IC 95% (Wilson); tracejado = 50%."
    >
      <div className="grid md:grid-cols-2 gap-4">
        {keys.map((k) => {
          const data: Row[] = metrics
            .filter((m) => m.eval[k])
            .map((m) => ({ timestep: m.timestep, [k]: m.eval[k].win_rate, [`${k}_ci`]: m.eval[k].ci95 }));
          const last = metrics.filter((m) => m.eval[k]).slice(-1)[0]?.eval[k];
          return (
            <div key={k} className="min-w-0">
              <div className="flex items-baseline justify-between text-xs mb-1">
                <span className="font-semibold text-zinc-200">
                  vs {playerLabel(k)}
                  {k === "prev" && <span className="text-zinc-500 font-normal"> (checkpoint da avaliação anterior)</span>}
                </span>
                {last && (
                  <span className="text-zinc-400 tabular-nums">
                    {pct(last.win_rate)} <span className="text-zinc-500">{ciText(last.ci95)}</span>
                  </span>
                )}
              </div>
              <MetricChart
                data={data}
                series={[{ key: k, label: playerLabel(k), color: opponentColor(k, keys) }]}
                bands
                yDomain={[0, 1]}
                yFormat={(v) => pct(v, 0)}
                refY={0.5}
                height={180}
              />
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function TrainVsEval({ metrics }: { metrics: RunMetric[] }) {
  const [view, setView] = useState<"reward" | "random" | "rule">("reward");
  const data: Row[] = metrics.map((m) => {
    if (view === "reward") {
      return {
        timestep: m.timestep,
        train: m.train.reward_mean,
        eval: m.eval_score == null ? null : 2 * m.eval_score - 1,
      };
    }
    return {
      timestep: m.timestep,
      train: m.train.by_opponent[view]?.win_rate ?? null,
      eval: m.eval[view]?.win_rate ?? null,
    };
  });
  const isReward = view === "reward";
  return (
    <Card
      title="Treino × avaliação"
      subtitle={
        isReward
          ? "Recompensa média por partida (+1 vitória, −1 derrota). Treino = contra a mistura da liga; avaliação = contra o conjunto fixo em mãos novas."
          : `Taxa de vitória contra ${playerLabel(view)}: partidas de treino × avaliação em mãos novas. Distância grande = sobreajuste.`
      }
      right={
        <div className="flex gap-1 text-xs" role="tablist">
          {(["reward", "random", "rule"] as const).map((v) => (
            <button
              key={v}
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
              className={`px-2 py-1 rounded ${view === v ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:text-zinc-100"}`}
            >
              {v === "reward" ? "Recompensa" : `vs ${playerLabel(v)}`}
            </button>
          ))}
        </div>
      }
    >
      <MetricChart
        data={data}
        series={[
          { key: "train", label: "Treino", color: SERIES[0] },
          { key: "eval", label: "Avaliação", color: SERIES[1] },
        ]}
        yDomain={isReward ? [-1, 1] : [0, 1]}
        yFormat={isReward ? (v) => v.toFixed(2) : (v) => pct(v, 0)}
        refY={isReward ? 0 : 0.5}
        height={240}
      />
    </Card>
  );
}

function BehaviourCharts({ metrics }: { metrics: RunMetric[] }) {
  const hasRule = metrics.some((m) => m.eval.rule);
  const evalKey = hasRule ? "rule" : "random";
  const entropy: Row[] = metrics.map((m) => ({ timestep: m.timestep, entropy: m.train.entropy }));
  const truco: Row[] = metrics.map((m) => ({
    timestep: m.timestep,
    train: m.train.truco_rate,
    eval: m.eval[evalKey]?.truco_rate ?? null,
  }));
  const run: Row[] = metrics.map((m) => ({
    timestep: m.timestep,
    train: m.train.run_rate,
    eval: m.eval[evalKey]?.run_rate ?? null,
  }));
  const pair = [
    { key: "train", label: "Treino", color: SERIES[0] },
    { key: "eval", label: `Aval. vs ${playerLabel(evalKey)}`, color: SERIES[1] },
  ];
  return (
    <section className="grid md:grid-cols-3 gap-4">
      <Card title="Entropia da política" subtitle="Cai conforme a política fica mais decidida.">
        <MetricChart
          data={entropy}
          series={[{ key: "entropy", label: "Entropia", color: SERIES[0] }]}
          yFormat={(v) => v.toFixed(2)}
          height={180}
        />
      </Card>
      <Card title="Taxa de truco" subtitle="Pedidos de truco do agente por mão.">
        <MetricChart data={truco} series={pair} yFormat={(v) => v.toFixed(2)} height={180} />
      </Card>
      <Card title="Taxa de corrida" subtitle="Fração das apostas enfrentadas em que o agente correu.">
        <MetricChart data={run} series={pair} yDomain={[0, 1]} yFormat={(v) => pct(v, 0)} height={180} />
      </Card>
    </section>
  );
}

function checkpointName(path: string): string {
  return path.replace(/\\/g, "/").split("/").pop() ?? path;
}

function Strength({ metrics }: { metrics: RunMetric[] }) {
  const { scoreData, eloData, anchors } = useMemo(() => {
    const scoreData: Row[] = metrics.map((m) => ({
      timestep: m.timestep,
      score: m.eval_score,
      score_ci: m.eval_score_ci95,
    }));
    // Pairwise results: every checkpoint vs the fixed opponents, plus each
    // checkpoint vs the previous one (links the checkpoints to each other).
    const pairs: PairResult[] = [];
    const stepOf = new Map<string, number>();
    for (const m of metrics) {
      const me = checkpointName(m.checkpoint);
      stepOf.set(me, m.timestep);
      for (const [opp, r] of Object.entries(m.eval)) {
        const wins = r.wins ?? Math.round(r.win_rate * r.games);
        const other = opp === "prev" ? checkpointName(r.opponent ?? "") : `opp:${opp}`;
        if (other) pairs.push({ a: me, b: other, wins, games: r.games });
      }
    }
    const elo = pairs.length ? fitElo(pairs, "opp:random") : {};
    const eloData: Row[] = metrics.map((m) => ({
      timestep: m.timestep,
      elo: elo[checkpointName(m.checkpoint)] ?? null,
    }));
    const anchors = Object.entries(elo)
      .filter(([k]) => k.startsWith("opp:") && k !== "opp:random")
      .map(([k, v]) => ({ name: playerLabel(k.slice(4)), elo: v }));
    return { scoreData, eloData, anchors };
  }, [metrics]);

  return (
    <section className="grid md:grid-cols-2 gap-4">
      <Card
        title="Força da IA por passo de treino"
        subtitle="Taxa de vitória somada contra o conjunto fixo de oponentes (sem o snapshot anterior), IC 95%."
      >
        <MetricChart
          data={scoreData}
          series={[{ key: "score", label: "Conjunto fixo", color: SERIES[0] }]}
          bands
          yDomain={[0, 1]}
          yFormat={(v) => pct(v, 0)}
          refY={0.5}
          height={220}
        />
      </Card>
      <Card
        title="Elo estimado"
        subtitle="Bradley–Terry sobre todos os confrontos da run; aleatório = 0. Referências na legenda abaixo."
      >
        <MetricChart
          data={eloData}
          series={[{ key: "elo", label: "Elo", color: SERIES[0] }]}
          yFormat={(v) => v.toFixed(0)}
          refY={0}
          height={220}
        />
        {anchors.length > 0 && (
          <div className="flex flex-wrap gap-3 text-xs text-zinc-400 mt-2">
            {anchors.map((a) => (
              <span key={a.name}>
                {a.name}: <span className="text-zinc-100 tabular-nums">{a.elo.toFixed(0)}</span>
              </span>
            ))}
          </div>
        )}
      </Card>
    </section>
  );
}

// --- Matrix + on-demand evaluation --------------------------------------------
function MatrixSection({ runId }: { runId: string }) {
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    getRunMatrix(runId)
      .then((m) => {
        setMatrix(m);
        setError(null);
      })
      .catch((e) => setError((e as Error).message));
  }, [runId]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <Card
      title="Matriz de confrontos"
      subtitle="Checkpoints (linhas) contra oponentes (colunas): taxa de vitória do checkpoint."
      right={<WinRateScale />}
    >
      {error && <ErrorBox error={error} onRetry={load} />}
      {!matrix && !error && <Loading />}
      {matrix && matrix.rows.length === 0 && <Empty>Sem avaliações ainda.</Empty>}
      {matrix && matrix.rows.length > 0 && (
        <div className="overflow-x-auto max-h-[28rem]">
          <table className="text-xs tabular-nums border-separate border-spacing-[2px]">
            <thead className="sticky top-0 bg-zinc-900 z-10">
              <tr>
                <th className="text-left font-medium text-zinc-400 pr-3 py-1">Checkpoint</th>
                {matrix.opponents.map((o) => (
                  <th key={o} className="font-medium text-zinc-400 px-2 py-1 min-w-[5.5rem]">
                    {playerLabel(o)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.rows.map((r) => (
                <tr key={r.checkpoint}>
                  <td className="pr-3 py-1 text-zinc-300 whitespace-nowrap">
                    {r.timestep != null ? steps(r.timestep) : playerLabel(r.checkpoint)}
                  </td>
                  {matrix.opponents.map((o) => {
                    const c = r.results[o];
                    return c ? (
                      <td
                        key={o}
                        className="text-center text-white font-semibold px-2 py-1.5 rounded-sm cursor-default"
                        style={{ background: winRateColor(c.win_rate) }}
                        title={`${playerLabel(r.checkpoint)} vs ${playerLabel(o)}: ${pct(c.win_rate)} em ${c.games} partidas (${ciText(c.ci95)})`}
                      >
                        {pct(c.win_rate, 0)}
                      </td>
                    ) : (
                      <td key={o} className="text-center text-zinc-600">—</td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <EvaluateForm runId={runId} onDone={load} />
    </Card>
  );
}

function EvaluateForm({ runId, onDone }: { runId: string; onDone: () => void }) {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[] | null>(null);
  const [players, setPlayers] = useState<PlayersList | null>(null);
  const [ckpt, setCkpt] = useState("");
  const [opp, setOpp] = useState("rule");
  const [games, setGames] = useState(500);
  const [job, setJob] = useState<ArenaJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getRunCheckpoints(runId), getArenaPlayers()])
      .then(([c, p]) => {
        setCheckpoints(c);
        setPlayers(p);
        setCkpt(c.find((x) => x.kind === "best")?.ref ?? c[0]?.ref ?? "");
      })
      .catch((e) => setError((e as Error).message));
  }, [runId]);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    const t = setInterval(async () => {
      try {
        const j = await getArenaJob(job.id);
        setJob(j);
        if (j.status === "done") onDone();
      } catch (e) {
        setError((e as Error).message);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [job, onDone]);

  async function submit() {
    setError(null);
    try {
      setJob(await createArenaJob({ a: ckpt, b: opp, games, run_id: runId }));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (!checkpoints || !players) return error ? <ErrorBox error={error} /> : null;
  const busy = job != null && (job.status === "queued" || job.status === "running");
  return (
    <div className="mt-4 border-t border-zinc-800 pt-4 space-y-3">
      <div className="text-xs text-zinc-400">Avaliar um checkpoint contra outro oponente (entra na matriz):</div>
      <div className="flex flex-wrap items-end gap-2 text-sm">
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Checkpoint
          <select value={ckpt} onChange={(e) => setCkpt(e.target.value)} className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-100 max-w-[16rem]">
            {checkpoints.map((c) => (
              <option key={c.ref} value={c.ref}>
                {c.kind === "snapshot" ? steps(c.timestep) : c.kind}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Oponente
          <select value={opp} onChange={(e) => setOpp(e.target.value)} className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-100 max-w-[16rem]">
            {players.builtin.map((b) => (
              <option key={b} value={b}>{playerLabel(b)}</option>
            ))}
            {players.models.map((m) => (
              <option key={m.ref} value={m.ref}>{playerLabel(m.ref)}</option>
            ))}
            {players.runs.flatMap((r) =>
              r.checkpoints.map((c) => (
                <option key={c.ref} value={c.ref}>{r.name} · {c.kind}</option>
              ))
            )}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Partidas
          <select value={games} onChange={(e) => setGames(Number(e.target.value))} className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-100">
            {[100, 500, 1000, 2000].map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </label>
        <Button tone="emerald" onClick={submit} disabled={busy || !ckpt}>
          {busy ? "Avaliando…" : "Avaliar"}
        </Button>
      </div>
      {error && <ErrorBox error={error} />}
      {job && <JobProgress job={job} />}
    </div>
  );
}
