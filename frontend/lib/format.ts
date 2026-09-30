export function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function ciText(ci: [number, number] | null | undefined): string {
  if (!ci) return "";
  return `IC95 ${pct(ci[0], 0)}–${pct(ci[1], 0)}`;
}

export function steps(v: number | null | undefined): string {
  if (v == null) return "—";
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(v % 1_000_000 ? 2 : 0)}M`;
  if (v >= 1_000) return `${Math.round(v / 1_000)}k`;
  return String(v);
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h) return `${h}h${String(m).padStart(2, "0")}`;
  if (m) return `${m}m${String(ss).padStart(2, "0")}s`;
  return `${ss}s`;
}

export function dateTime(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "—";
  return new Date(epochSeconds * 1000).toLocaleString("pt-BR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const OPPONENT_LABELS: Record<string, string> = {
  random: "Aleatório",
  rule: "Regra",
  prev: "Snapshot anterior",
  snapshot: "Snapshots",
  latest: "Mais recente",
  facil: "Fácil",
  medio: "Médio",
  impossivel: "Impossível",
};

/** Human label for an opponent key or player reference. */
export function playerLabel(ref: string): string {
  if (OPPONENT_LABELS[ref]) return OPPONENT_LABELS[ref];
  const m = ref.match(/^ppo(?:-stoch)?:(.+)$/);
  if (!m) return ref;
  const parts = m[1].replace(/\\/g, "/").split("/");
  const file = parts[parts.length - 1].replace(/\.zip$/, "");
  if (parts[0] === "runs" && parts.length >= 3) {
    const run = parts[1].replace(/^\d{8}-\d{6}-/, "");
    const step = file.match(/^step_0*(\d+)$/);
    return `${run} · ${step ? steps(Number(step[1])) : file}`;
  }
  return `PPO ${file.replace(/^truco_ppo_?/, "") || "base"}`;
}
