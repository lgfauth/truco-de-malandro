"use client";

import { StatusBadge } from "@/components/ui";
import { ciText, pct, playerLabel } from "@/lib/format";
import type { ArenaJob } from "@/types/runs";

export function JobProgress({ job }: { job: ArenaJob }) {
  const frac = job.progress.total ? job.progress.done / job.progress.total : 0;
  return (
    <div className="text-xs text-zinc-300 space-y-1">
      <div className="flex justify-between gap-2">
        <span>
          {playerLabel(job.a)} vs {playerLabel(job.b)} · <StatusBadge status={job.status} />
        </span>
        <span className="tabular-nums">{job.progress.done}/{job.progress.total}</span>
      </div>
      <div className="h-1.5 rounded bg-zinc-800 overflow-hidden" role="progressbar" aria-valuenow={Math.round(frac * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-emerald-500 transition-all" style={{ width: `${frac * 100}%` }} />
      </div>
      {job.status === "done" && job.result && (
        <div>
          Resultado: <span className="font-semibold text-zinc-100">{pct(job.result.win_rate_a)}</span>{" "}
          <span className="text-zinc-500">{ciText(job.result.ci95)}</span>
        </div>
      )}
      {job.status === "failed" && <div className="text-rose-300">Falhou: {job.error}</div>}
    </div>
  );
}
