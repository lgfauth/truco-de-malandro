"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { MetricChart, SERIES, opponentColor, type Row } from "@/components/charts";
import { Button, Card, Empty, ErrorBox, Page, Stat } from "@/components/ui";
import {
  getRunMetrics,
  getTrainStatus,
  metricsWebSocketUrl,
  pauseTrain,
  resetTrain,
  startTrain,
} from "@/lib/api";
import { duration, pct, playerLabel, steps } from "@/lib/format";
import type { TrainStatus } from "@/types/game";
import type { RunMetric } from "@/types/runs";

type WSMsg =
  | { type: "metrics"; run_id: string | null; full: RunMetric[] }
  | ({ type: "status" } & TrainStatus);

export default function WatchPage() {
  const [metrics, setMetrics] = useState<RunMetric[]>([]);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalSteps, setTotalSteps] = useState(500_000);
  const [obsVersion, setObsVersion] = useState<"v1" | "v2">("v2");
  const [busy, setBusy] = useState(false);
  const runIdRef = useRef<string | null>(null);

  // Follow the current run: reload its metrics from disk when it changes
  // (covers page reloads and runs started elsewhere).
  function followRun(runId: string | null | undefined) {
    if (!runId || runId === runIdRef.current) return;
    runIdRef.current = runId;
    getRunMetrics(runId)
      .then((m) => setMetrics(m.items))
      .catch(() => {});
  }

  useEffect(() => {
    const tick = () =>
      getTrainStatus()
        .then((s) => {
          setStatus(s);
          followRun(s.run_id);
          setError(null);
        })
        .catch((e) => setError(`Backend indisponível: ${(e as Error).message}`));
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const ws = new WebSocket(metricsWebSocketUrl());
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);
    ws.onmessage = (ev) => {
      try {
        const msg: WSMsg = JSON.parse(ev.data);
        if (msg.type === "metrics" && msg.full?.length) {
          if (msg.run_id && msg.run_id !== runIdRef.current) {
            runIdRef.current = msg.run_id;
            setMetrics(msg.full);
          } else {
            setMetrics((prev) => {
              const seen = new Set(prev.map((m) => m.timestep));
              return [...prev, ...msg.full.filter((m) => !seen.has(m.timestep))];
            });
          }
        } else if (msg.type === "status") {
          const { type: _t, ...rest } = msg;
          setStatus(rest as TrainStatus);
        }
      } catch {
        /* ignore malformed frames */
      }
    };
    return () => ws.close();
  }, []);

  async function act(fn: () => Promise<unknown>) {
    setError(null);
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const keys = useMemo(() => {
    const k: string[] = [];
    for (const m of metrics) for (const key of Object.keys(m.eval)) if (!k.includes(key)) k.push(key);
    return k;
  }, [metrics]);

  const winData: Row[] = metrics.map((m) => {
    const row: Row = { timestep: m.timestep };
    for (const k of keys) row[k] = m.eval[k]?.win_rate ?? null;
    return row;
  });
  const entropy: Row[] = metrics.map((m) => ({ timestep: m.timestep, entropy: m.train.entropy }));
  const truco: Row[] = metrics.map((m) => ({
    timestep: m.timestep,
    train: m.train.truco_rate,
    run: m.train.run_rate,
  }));

  const running = !!status?.running;
  const progress = status?.total_timesteps ? status.timestep / status.total_timesteps : 0;
  const latest = metrics[metrics.length - 1];

  return (
    <Page
      title="Treino do agente"
      subtitle={
        <>
          Treino em liga dentro do servidor. Para treinos longos prefira{" "}
          <code className="text-zinc-300">python -m agent.train</code> no backend. WS:{" "}
          {wsConnected ? "conectado" : "desconectado"}
        </>
      }
    >
      <Card title="Controle">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-xs text-zinc-400">
            Passos
            <select
              value={totalSteps}
              onChange={(e) => setTotalSteps(Number(e.target.value))}
              disabled={running}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100"
            >
              {[100_000, 500_000, 1_000_000, 3_000_000].map((n) => (
                <option key={n} value={n}>{steps(n)}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-zinc-400">
            Observação
            <select
              value={obsVersion}
              onChange={(e) => setObsVersion(e.target.value as "v1" | "v2")}
              disabled={running}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100"
            >
              <option value="v2">v2 (força, manilha, histórico)</option>
              <option value="v1">v1 (original)</option>
            </select>
          </label>
          <Button
            tone="emerald"
            disabled={running || busy}
            onClick={() => act(() => startTrain({ total_timesteps: totalSteps, obs_version: obsVersion, name: `web_${obsVersion}` }))}
          >
            Iniciar treino
          </Button>
          <Button tone="amber" disabled={!running || busy} onClick={() => act(pauseTrain)}>
            Parar
          </Button>
          <Button
            tone="rose"
            disabled={busy}
            onClick={() =>
              act(async () => {
                await resetTrain();
                runIdRef.current = null;
                setMetrics([]);
              })
            }
          >
            Limpar
          </Button>
        </div>
        <p className="text-xs text-zinc-500 mt-2">
          Parar encerra a run e salva o modelo (final.zip); Limpar só esquece o estado desta tela —
          as runs continuam em <Link href="/runs" className="underline">Runs</Link>.
        </p>
      </Card>

      {error && <ErrorBox error={error} />}
      {status?.error && <ErrorBox error={`O treino falhou: ${status.error}`} />}

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat label="Status" value={running ? "Rodando" : status?.paused ? "Parado" : "Ocioso"} />
        <Stat
          label="Passos"
          value={steps(status?.timestep ?? 0)}
          hint={status?.total_timesteps ? `${pct(progress, 0)} de ${steps(status.total_timesteps)}` : undefined}
        />
        <Stat label="Partidas de treino" value={status?.episode ?? 0} />
        <Stat
          label="Run"
          value={
            status?.run_id ? (
              <Link href={`/runs/${encodeURIComponent(status.run_id)}`} className="underline text-base">
                ver detalhes
              </Link>
            ) : "—"
          }
          hint={latest ? `última aval. em ${steps(latest.timestep)} · ${duration(latest.elapsed_s)}` : undefined}
        />
      </section>

      {metrics.length === 0 ? (
        <Empty>
          {running
            ? "Treinando… a primeira avaliação aparece no primeiro intervalo de avaliação."
            : "Nenhum treino acompanhado agora. Inicie um acima ou abra uma run em Runs."}
        </Empty>
      ) : (
        <>
          <Card title="Taxa de vitória na avaliação" subtitle="Mãos fixas, dois lados; tracejado = 50%.">
            <MetricChart
              data={winData}
              series={keys.map((k) => ({ key: k, label: playerLabel(k), color: opponentColor(k, keys) }))}
              yDomain={[0, 1]}
              yFormat={(v) => pct(v, 0)}
              refY={0.5}
              height={260}
            />
          </Card>
          <section className="grid md:grid-cols-2 gap-4">
            <Card title="Entropia da política">
              <MetricChart
                data={entropy}
                series={[{ key: "entropy", label: "Entropia", color: SERIES[0] }]}
                height={180}
              />
            </Card>
            <Card title="Truco e corrida no treino" subtitle="Trucos por mão · fração das apostas em que correu.">
              <MetricChart
                data={truco}
                series={[
                  { key: "train", label: "Trucos/mão", color: SERIES[0] },
                  { key: "run", label: "Corrida", color: SERIES[1] },
                ]}
                yDomain={[0, 1]}
                height={180}
              />
            </Card>
          </section>
        </>
      )}
    </Page>
  );
}
