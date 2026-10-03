"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { MetricChart, SERIES, type Row, type Series } from "@/components/charts";
import { Card, Empty, ErrorBox, Loading, Page, StatusBadge } from "@/components/ui";
import { getRunMetrics, listRuns } from "@/lib/api";
import { dateTime, duration, pct, playerLabel, steps } from "@/lib/format";
import type { RunMetric, RunSummary } from "@/types/runs";

const MAX_COMPARE = 4;

type Selected = { id: string; slot: number };

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Selected[]>([]);
  const [metrics, setMetrics] = useState<Record<string, RunMetric[]>>({});
  const [metricKey, setMetricKey] = useState("eval_score");

  const load = useCallback(() => {
    setError(null);
    listRuns()
      .then(setRuns)
      .catch((e) => setError((e as Error).message));
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [load]);

  // Fetch metrics of newly selected runs.
  useEffect(() => {
    for (const { id } of selected) {
      if (metrics[id]) continue;
      getRunMetrics(id)
        .then((m) => setMetrics((prev) => ({ ...prev, [id]: m.items })))
        .catch((e) => setError((e as Error).message));
    }
  }, [selected, metrics]);

  function toggle(id: string) {
    setSelected((prev) => {
      if (prev.some((s) => s.id === id)) return prev.filter((s) => s.id !== id);
      if (prev.length >= MAX_COMPARE) return prev;
      // Lowest free slot, so survivors keep their color.
      const used = new Set(prev.map((s) => s.slot));
      let slot = 0;
      while (used.has(slot)) slot++;
      return [...prev, { id, slot }];
    });
  }

  const opponentKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const { id } of selected)
      for (const m of metrics[id] ?? []) Object.keys(m.eval).forEach((k) => keys.add(k));
    return Array.from(keys).filter((k) => k !== "prev");
  }, [selected, metrics]);

  const { data, series } = useMemo(() => {
    const series: Series[] = selected.map(({ id, slot }) => ({
      key: id,
      label: runs?.find((r) => r.id === id)?.name ?? id,
      color: SERIES[slot],
    }));
    const byStep = new Map<number, Row>();
    for (const { id } of selected) {
      for (const m of metrics[id] ?? []) {
        const row = byStep.get(m.timestep) ?? { timestep: m.timestep };
        if (metricKey === "eval_score") {
          row[id] = m.eval_score;
          if (m.eval_score_ci95) row[`${id}_ci`] = m.eval_score_ci95;
        } else if (m.eval[metricKey]) {
          row[id] = m.eval[metricKey].win_rate;
          row[`${id}_ci`] = m.eval[metricKey].ci95;
        }
        byStep.set(m.timestep, row);
      }
    }
    const data = Array.from(byStep.values()).sort(
      (a, b) => (a.timestep as number) - (b.timestep as number)
    );
    return { data, series };
  }, [selected, metrics, metricKey, runs]);

  return (
    <Page
      title="Runs de treino"
      subtitle="Cada execução de treino grava config, métricas e checkpoints em runs/<id>/."
    >
      {error && <ErrorBox error={error} onRetry={load} />}
      {!runs && !error && <Loading />}
      {runs && runs.length === 0 && (
        <Empty>
          Nenhuma run ainda. Rode <code className="text-zinc-200">python -m agent.train</code> no
          backend ou inicie um treino em <Link href="/watch" className="underline">Treino</Link>.
        </Empty>
      )}

      {runs && runs.length > 0 && (
        <Card
          title="Execuções"
          subtitle={`Marque até ${MAX_COMPARE} para comparar no gráfico abaixo.`}
        >
          <div className="overflow-x-auto -mx-4 px-4">
            <table className="w-full text-sm min-w-[720px]">
              <thead className="text-xs text-zinc-400 text-left">
                <tr>
                  <th className="py-2 pr-2 w-8" />
                  <th className="py-2 pr-3 font-medium">Run</th>
                  <th className="py-2 pr-3 font-medium">Status</th>
                  <th className="py-2 pr-3 font-medium">Config</th>
                  <th className="py-2 pr-3 font-medium text-right">Passos</th>
                  <th className="py-2 pr-3 font-medium text-right">Duração</th>
                  <th className="py-2 pr-3 font-medium text-right">Melhor checkpoint</th>
                  <th className="py-2 font-medium text-right">Última aval.</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => {
                  const sel = selected.find((s) => s.id === r.id);
                  const disabled = !sel && selected.length >= MAX_COMPARE;
                  return (
                    <tr key={r.id} className="border-t border-zinc-800 align-top">
                      <td className="py-2 pr-2">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={!!sel}
                            disabled={disabled}
                            onChange={() => toggle(r.id)}
                            aria-label={`Comparar ${r.name}`}
                            className="h-4 w-4 accent-emerald-500"
                          />
                          {sel && (
                            <span
                              className="ml-1.5 inline-block h-2.5 w-2.5 rounded-full"
                              style={{ background: SERIES[sel.slot] }}
                            />
                          )}
                        </label>
                      </td>
                      <td className="py-2 pr-3">
                        <Link href={`/runs/${encodeURIComponent(r.id)}`} className="font-semibold text-zinc-100 hover:underline">
                          {r.name}
                        </Link>
                        <div className="text-xs text-zinc-500">{dateTime(r.started_at)} · {r.id}</div>
                      </td>
                      <td className="py-2 pr-3"><StatusBadge status={r.status} /></td>
                      <td className="py-2 pr-3 text-xs text-zinc-300">
                        obs {r.config.obs_version ?? r.obs_version} · seed {r.config.seed ?? "—"}
                        <div className="text-zinc-500">
                          {r.config.league
                            ? Object.entries(r.config.league).map(([k, v]) => `${playerLabel(k)} ${Math.round(v * 100)}%`).join(" · ")
                            : "—"}
                        </div>
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums">
                        {steps(r.timesteps)}
                        <span className="text-zinc-500"> / {steps(r.total_timesteps)}</span>
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums">{duration(r.duration_s)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">
                        {r.best ? (
                          <>
                            {pct(r.best.score)}
                            <div className="text-xs text-zinc-500">em {steps(r.best.timestep)}</div>
                          </>
                        ) : "—"}
                      </td>
                      <td className="py-2 text-right text-xs tabular-nums text-zinc-300">
                        {r.last_eval
                          ? Object.entries(r.last_eval.win_rates)
                              .filter(([k]) => k !== "prev")
                              .map(([k, v]) => (
                                <div key={k}>
                                  {playerLabel(k)} <span className="text-zinc-100">{pct(v, 0)}</span>
                                </div>
                              ))
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-zinc-500 mt-2">
            Melhor checkpoint = maior taxa de vitória somada contra os oponentes fixos da avaliação
            (tudo menos o snapshot anterior).
          </p>
        </Card>
      )}

      {selected.length > 0 && (
        <Card
          title="Comparação de runs"
          subtitle="Taxa de vitória na avaliação (mãos fixas, dois lados), com banda de IC 95%."
          right={
            <select
              value={metricKey}
              onChange={(e) => setMetricKey(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs"
              aria-label="Métrica"
            >
              <option value="eval_score">Contra o conjunto fixo</option>
              {opponentKeys.map((k) => (
                <option key={k} value={k}>
                  Contra {playerLabel(k)}
                </option>
              ))}
            </select>
          }
        >
          {selected.some(({ id }) => !metrics[id]) ? (
            <Loading label="Carregando métricas…" />
          ) : (
            <MetricChart
              data={data}
              series={series}
              bands
              yDomain={[0, 1]}
              yFormat={(v) => pct(v, 0)}
              refY={0.5}
              height={280}
            />
          )}
        </Card>
      )}
    </Page>
  );
}
