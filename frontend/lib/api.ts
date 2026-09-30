import type { GameStateDTO, TrainStatus } from "@/types/game";
import type {
  ArenaJob,
  CheckpointInfo,
  Matrix,
  PlayersList,
  RunMetric,
  RunSummary,
} from "@/types/runs";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const isGet = !init?.method || init.method === "GET";
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      ...(isGet ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// --- Game --------------------------------------------------------
export function newGame(opponent?: string) {
  return request<GameStateDTO>("/game/new", {
    method: "POST",
    body: JSON.stringify(opponent ? { opponent } : {}),
  });
}

export function sendAction(game_id: string, action: number) {
  return request<GameStateDTO>("/game/action", {
    method: "POST",
    body: JSON.stringify({ game_id, action }),
  });
}

export function nextHand(game_id: string) {
  return request<GameStateDTO>("/game/next-hand", {
    method: "POST",
    body: JSON.stringify({ game_id }),
  });
}

export function getState(game_id: string) {
  return request<GameStateDTO>(
    `/game/state?game_id=${encodeURIComponent(game_id)}`
  );
}

// --- Training ----------------------------------------------------
export interface StartTrainOptions {
  total_timesteps?: number;
  name?: string;
  obs_version?: "v1" | "v2";
  eval_every?: number;
  eval_games?: number;
}

export function startTrain(opts: StartTrainOptions = {}) {
  return request<{ status: string }>("/train/start", {
    method: "POST",
    body: JSON.stringify({ total_timesteps: 500000, ...opts }),
  });
}

export function pauseTrain() {
  return request<{ status: string } & TrainStatus>("/train/pause", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function resetTrain() {
  return request<{ status: string }>("/train/reset", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function getTrainStatus() {
  return request<TrainStatus>("/train/status");
}

export function metricsWebSocketUrl(): string {
  return API_URL.replace(/^http/, "ws") + "/ws/metrics";
}

// --- Runs --------------------------------------------------------
export function listRuns() {
  return request<RunSummary[]>("/runs");
}

export function getRun(id: string) {
  return request<{ run: RunSummary; config: Record<string, unknown> }>(
    `/runs/${encodeURIComponent(id)}`
  );
}

export function getRunMetrics(id: string, since = 0) {
  return request<{ items: RunMetric[]; next: number }>(
    `/runs/${encodeURIComponent(id)}/metrics?since=${since}`
  );
}

export function getRunCheckpoints(id: string) {
  return request<CheckpointInfo[]>(`/runs/${encodeURIComponent(id)}/checkpoints`);
}

export function getRunMatrix(id: string) {
  return request<Matrix>(`/runs/${encodeURIComponent(id)}/matrix`);
}

// --- Arena -------------------------------------------------------
export function getArenaPlayers() {
  return request<PlayersList>("/arena/players");
}

export function createArenaJob(body: {
  a: string;
  b: string;
  games: number;
  seed?: number;
  run_id?: string;
}) {
  return request<ArenaJob>("/arena/jobs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getArenaJob(id: string) {
  return request<ArenaJob>(`/arena/jobs/${encodeURIComponent(id)}`);
}

export function listArenaJobs(limit = 20) {
  return request<ArenaJob[]>(`/arena/jobs?limit=${limit}`);
}
