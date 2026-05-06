"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getTrainStatus,
  metricsWebSocketUrl,
  pauseTrain,
  resetTrain,
  startTrain,
} from "@/lib/api";
import type { EvalMetric, TrainStatus, WSMessage } from "@/types/game";

function formatElapsed(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const ss = (s % 60).toString().padStart(2, "0");
  return `${m}:${ss}`;
}

export default function WatchPage() {
  const [metrics, setMetrics] = useState<EvalMetric[]>([]);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const startedAtRef = useRef<number | null>(null);
  const lastTimestepRef = useRef<number>(0);

  // Initial fetch
  useEffect(() => {
    getTrainStatus()
      .then((s) => {
        setStatus(s);
        if (s.latest_metric) setMetrics([s.latest_metric]);
      })
      .catch(() => {});
  }, []);

  // HTTP polling — runs always every 3s. When WS is connected, only updates
  // status (WS already pushes metrics). When WS is down, also accumulates metrics.
  useEffect(() => {
    const id = setInterval(() => {
      getTrainStatus()
        .then((s) => {
          setStatus(s);
          if (!wsConnected && s.latest_metric) {
            setMetrics((prev) => {
              const last = prev[prev.length - 1];
              if (!last || last.timestep !== s.latest_metric!.timestep) {
                return [...prev, s.latest_metric!];
              }
              return prev;
            });
          }
        })
        .catch(() => {});
    }, 3000);
    return () => clearInterval(id);
  }, [wsConnected]);

  useEffect(() => {
    const ws = new WebSocket(metricsWebSocketUrl());
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);
    ws.onmessage = (ev) => {
      try {
        const msg: WSMessage = JSON.parse(ev.data);
        if (msg.type === "metrics") {
          setMetrics((prev) => [...prev, ...msg.items]);
        } else if (msg.type === "status") {
          const { type: _t, ...rest } = msg;
          setStatus(rest as TrainStatus);
        }
      } catch {
        /* ignore */
      }
    };
    return () => ws.close();
  }, []);

  // Track elapsed wall-clock time of the current training run.
  // Anchor when running flips on; freeze when off; zero out when /train/reset
  // drops timestep back to 0.
  useEffect(() => {
    if (!status) return;
    if (status.running) {
      if (startedAtRef.current == null) {
        startedAtRef.current = Date.now() - elapsed * 1000;
      }
    } else {
      startedAtRef.current = null;
    }
    if (status.timestep === 0 && lastTimestepRef.current > 0) {
      setElapsed(0);
      startedAtRef.current = status.running ? Date.now() : null;
    }
    lastTimestepRef.current = status.timestep;
  }, [status, elapsed]);

  useEffect(() => {
    if (!status?.running) return;
    const id = setInterval(() => {
      if (startedAtRef.current != null) {
        setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
      }
    }, 1000);
    return () => clearInterval(id);
  }, [status?.running]);

  async function handleStart() {
    setError(null);
    try {
      await startTrain();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handlePause() {
    setError(null);
    try {
      await pauseTrain();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handleReset() {
    setError(null);
    try {
      await resetTrain();
      setMetrics([]);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const chartData = metrics.map((m) => ({
    episode: m.episode,
    timestep: m.timestep,
    win_rate: Number((m.win_rate * 100).toFixed(2)),
    mean_reward: Number(m.mean_reward.toFixed(3)),
    mean_steps: Number(m.mean_steps.toFixed(2)),
    entropy: Number((m.entropy ?? 0).toFixed(4)),
    truco_rate_pct: Number(((m.truco_rate ?? 0) * 100).toFixed(2)),
    run_rate_pct: Number(((m.run_rate ?? 0) * 100).toFixed(2)),
  }));

  return (
    <main className="min-h-screen px-6 py-8 max-w-6xl mx-auto space-y-6">
      <nav className="flex items-center justify-between">
        <Link href="/" className="text-zinc-400 hover:text-zinc-100">
          ← Voltar
        </Link>
        <span className="text-xs text-zinc-500">
          WS: {wsConnected ? "conectado" : "desconectado"}
        </span>
      </nav>

      <h1 className="text-3xl font-bold">Treino do agente</h1>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleStart}
          className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 font-semibold"
        >
          Iniciar treino
        </button>
        <button
          onClick={handlePause}
          className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 font-semibold"
        >
          Pausar
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 rounded-lg bg-rose-700 hover:bg-rose-600 font-semibold"
        >
          Resetar
        </button>
      </div>

      {error && (
        <div className="p-3 rounded bg-rose-950 border border-rose-800 text-rose-200 text-sm">
          {error}
        </div>
      )}

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCard
          label="Status"
          value={status?.running ? "Rodando" : status?.paused ? "Pausado" : "Parado"}
          tone={status?.running ? "emerald" : "zinc"}
        />
        <StatusCard label="Time" value={formatElapsed(elapsed)} />
        <StatusCard label="Timestep" value={status?.timestep ?? 0} />
        <StatusCard label="Episode" value={status?.episode ?? 0} />
      </section>

      <section className="grid grid-cols-2 gap-4">
        <ChartCard title="Win rate (%)">
          <MiniChart data={chartData} dataKey="win_rate" stroke="#10b981" yDomain={[0, 100]} />
        </ChartCard>
        <ChartCard title="Steps médios">
          <MiniChart data={chartData} dataKey="mean_steps" stroke="#3b82f6" />
        </ChartCard>
        <ChartCard title="Entropy da policy">
          <MiniChart data={chartData} dataKey="entropy" stroke="#8b5cf6" />
        </ChartCard>
        <ChartCard title="Taxa de truco (%)">
          <MiniChart data={chartData} dataKey="truco_rate_pct" stroke="#f59e0b" yDomain={[0, 100]} />
        </ChartCard>
      </section>

      {status?.latest_metric && (
        <section className="text-sm text-zinc-400">
          Última avaliação: win rate{" "}
          <span className="text-zinc-100 font-semibold">
            {(status.latest_metric.win_rate * 100).toFixed(1)}%
          </span>{" "}
          · steps médios{" "}
          <span className="text-zinc-100 font-semibold">
            {status.latest_metric.mean_steps.toFixed(1)}
          </span>
        </section>
      )}
    </main>
  );
}

function StatusCard({
  label,
  value,
  tone = "zinc",
}: {
  label: string;
  value: string | number;
  tone?: "zinc" | "emerald";
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <div className="text-xs uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div
        className={`mt-1 text-2xl font-bold ${
          tone === "emerald" ? "text-emerald-400" : "text-zinc-100"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function MiniChart({
  data,
  dataKey,
  stroke,
  yDomain,
}: {
  data: Array<Record<string, number>>;
  dataKey: string;
  stroke: string;
  yDomain?: [number, number];
}) {
  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={data}>
        <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
        <XAxis dataKey="episode" stroke="#71717a" />
        <YAxis domain={yDomain ?? ["auto", "auto"]} stroke="#71717a" />
        <Tooltip
          contentStyle={{
            background: "#18181b",
            border: "1px solid #3f3f46",
          }}
        />
        <Line type="monotone" dataKey={dataKey} stroke={stroke} strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <h2 className="text-sm font-semibold text-zinc-300 mb-3">{title}</h2>
      {children}
    </div>
  );
}
