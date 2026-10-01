"use client";

import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { steps } from "@/lib/format";

// Categorical slots, dark steps (validated on the #18181b card surface).
// Assigned in fixed order, never cycled; color follows the entity.
export const SERIES = [
  "#3987e5", // blue
  "#d95926", // orange
  "#199e70", // aqua
  "#c98500", // yellow
  "#d55181", // magenta
  "#008300", // green
  "#9085e9", // violet
  "#e66767", // red
];

export const INK = {
  primary: "#ffffff",
  secondary: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  axis: "#383835",
};

// Evaluation opponents keep the same color on every chart.
const OPPONENT_SLOT: Record<string, number> = {
  random: 0,
  rule: 1,
  prev: 2,
  "ppo:truco_ppo_1M": 3,
};

export function opponentColor(key: string, all: string[]): string {
  if (key in OPPONENT_SLOT) return SERIES[OPPONENT_SLOT[key]];
  const others = all.filter((k) => !(k in OPPONENT_SLOT)).sort();
  return SERIES[4 + (others.indexOf(key) % 4)];
}

export interface Series {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
}

export type Row = Record<string, number | [number, number] | null | undefined | string>;

export function Legend({ series }: { series: Series[] }) {
  if (series.length < 2) return null;
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-300 mb-2">
      {series.map((s) => (
        <li key={s.key} className="flex items-center gap-1.5">
          <svg width="16" height="8" aria-hidden>
            <line
              x1="0"
              y1="4"
              x2="16"
              y2="4"
              stroke={s.color}
              strokeWidth="2"
              strokeDasharray={s.dashed ? "4 3" : undefined}
            />
          </svg>
          {s.label}
        </li>
      ))}
    </ul>
  );
}

function ChartTooltip({
  active,
  payload,
  series,
  xKey,
  xFormat,
  yFormat,
}: {
  active?: boolean;
  payload?: { payload: Row }[];
  series: Series[];
  xKey: string;
  xFormat: (v: number) => string;
  yFormat: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-lg border border-white/10 bg-zinc-900/95 px-3 py-2 text-xs shadow-lg">
      <div className="text-zinc-400 mb-1">{xFormat(row[xKey] as number)}</div>
      {series.map((s) => {
        const v = row[s.key];
        if (v == null || typeof v !== "number") return null;
        const ci = row[`${s.key}_ci`];
        return (
          <div key={s.key} className="flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
            <span className="text-zinc-300">{s.label}</span>
            <span className="ml-auto pl-3 font-semibold text-zinc-100 tabular-nums">{yFormat(v)}</span>
            {Array.isArray(ci) && (
              <span className="text-zinc-500 tabular-nums">
                [{yFormat(ci[0])}–{yFormat(ci[1])}]
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * Lines over training steps, optional confidence bands (`<key>_ci` = [lo, hi]),
 * crosshair tooltip, direct end labels (<= 4 series) and a table view.
 */
export function MetricChart({
  data,
  series,
  xKey = "timestep",
  yDomain,
  yFormat = (v) => v.toFixed(2),
  xFormat = (v) => `${steps(v)} passos`,
  refY,
  height = 220,
  bands = false,
}: {
  data: Row[];
  series: Series[];
  xKey?: string;
  yDomain?: [number | "auto", number | "auto"];
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  refY?: number;
  height?: number;
  bands?: boolean;
}) {
  const [showTable, setShowTable] = useState(false);
  const directLabels = series.length > 1 && series.length <= 4;

  // End-of-line labels, nudged apart in value space so they never overlap.
  const endLabels = useMemo(() => {
    if (!directLabels || !data.length) return [];
    const ends = series
      .map((s) => {
        for (let i = data.length - 1; i >= 0; i--) {
          const v = data[i][s.key];
          if (typeof v === "number") return { s, x: data[i][xKey] as number, v };
        }
        return null;
      })
      .filter(Boolean) as { s: Series; x: number; v: number }[];
    const values = data.flatMap((r) => series.map((s) => r[s.key]).filter((v) => typeof v === "number")) as number[];
    const lo = typeof yDomain?.[0] === "number" ? yDomain[0] : Math.min(...values);
    const hi = typeof yDomain?.[1] === "number" ? yDomain[1] : Math.max(...values);
    const gap = (hi - lo || 1) * 0.09;
    ends.sort((a, b) => a.v - b.v);
    const placed: number[] = [];
    ends.forEach((e, i) => {
      placed[i] = i === 0 ? e.v : Math.max(e.v, placed[i - 1] + gap);
    });
    return ends.map((e, i) => ({ ...e, y: placed[i] }));
  }, [data, series, xKey, yDomain, directLabels]);

  if (!data.length) {
    return <p className="text-sm text-zinc-500 py-8 text-center">Sem pontos ainda.</p>;
  }

  return (
    <div>
      <div className="flex items-start justify-between gap-2">
        <Legend series={series} />
        <button
          onClick={() => setShowTable((v) => !v)}
          className="ml-auto text-[11px] text-zinc-500 hover:text-zinc-200 underline-offset-2 hover:underline shrink-0"
        >
          {showTable ? "ver gráfico" : "ver tabela"}
        </button>
      </div>
      {showTable ? (
        <div className="overflow-x-auto max-h-72">
          <table className="w-full text-xs tabular-nums">
            <thead className="text-zinc-400 sticky top-0 bg-zinc-900">
              <tr>
                <th className="text-left font-medium py-1 pr-3">Passos</th>
                {series.map((s) => (
                  <th key={s.key} className="text-right font-medium py-1 px-2">
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-zinc-200">
              {data.map((r, i) => (
                <tr key={i} className="border-t border-zinc-800">
                  <td className="py-1 pr-3">{steps(r[xKey] as number)}</td>
                  {series.map((s) => {
                    const v = r[s.key];
                    const ci = r[`${s.key}_ci`];
                    return (
                      <td key={s.key} className="text-right py-1 px-2">
                        {typeof v === "number" ? yFormat(v) : "—"}
                        {Array.isArray(ci) && (
                          <span className="text-zinc-500"> [{yFormat(ci[0])}–{yFormat(ci[1])}]</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data} margin={{ top: 8, right: directLabels ? 84 : 12, bottom: 0, left: -8 }}>
            <CartesianGrid stroke={INK.grid} vertical={false} />
            <XAxis
              dataKey={xKey}
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) => steps(v)}
              stroke={INK.axis}
              tick={{ fill: INK.muted, fontSize: 11 }}
              tickLine={false}
            />
            <YAxis
              domain={yDomain ?? ["auto", "auto"]}
              tickFormatter={yFormat}
              stroke={INK.axis}
              tick={{ fill: INK.muted, fontSize: 11 }}
              tickLine={false}
              width={52}
            />
            {refY != null && <ReferenceLine y={refY} stroke={INK.muted} strokeDasharray="4 4" />}
            <Tooltip
              cursor={{ stroke: INK.muted, strokeWidth: 1 }}
              content={<ChartTooltip series={series} xKey={xKey} xFormat={xFormat} yFormat={yFormat} />}
            />
            {bands &&
              series.map((s) => (
                <Area
                  key={`${s.key}_band`}
                  dataKey={`${s.key}_ci`}
                  stroke="none"
                  fill={s.color}
                  fillOpacity={0.14}
                  isAnimationActive={false}
                  activeDot={false}
                  connectNulls
                />
              ))}
            {series.map((s) => (
              <Line
                key={s.key}
                dataKey={s.key}
                stroke={s.color}
                strokeWidth={2}
                strokeDasharray={s.dashed ? "5 4" : undefined}
                dot={false}
                activeDot={{ r: 4, stroke: "#18181b", strokeWidth: 2 }}
                isAnimationActive={false}
                connectNulls
              />
            ))}
            {endLabels.map((e) => (
              <ReferenceDot
                key={`lbl_${e.s.key}`}
                x={e.x}
                y={e.y}
                r={0}
                ifOverflow="visible"
                label={{ value: e.s.label, position: "right", fill: INK.secondary, fontSize: 11 }}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// --- Heatmap (win rate, diverging around 50%) ---------------------------------
const MID = [0x38, 0x38, 0x35];
const WIN = [0x25, 0x6a, 0xbf];
const LOSS = [0xb9, 0x3a, 0x3a];

export function winRateColor(v: number): string {
  const t = Math.max(-1, Math.min(1, (v - 0.5) / 0.4));
  const end = t >= 0 ? WIN : LOSS;
  const a = Math.abs(t);
  const c = MID.map((m, i) => Math.round(m + (end[i] - m) * a));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function WinRateScale() {
  return (
    <div className="flex items-center gap-2 text-[11px] text-zinc-400">
      <span>perde</span>
      <span
        className="h-2 w-32 rounded"
        style={{
          background: `linear-gradient(90deg, ${winRateColor(0.1)}, ${winRateColor(0.5)}, ${winRateColor(0.9)})`,
        }}
      />
      <span>vence</span>
      <span className="text-zinc-500">(50% = empate de forças)</span>
    </div>
  );
}
