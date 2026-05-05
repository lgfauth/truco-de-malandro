import type { GameStateDTO, TrainStatus } from "@/types/game";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
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
export function newGame() {
  return request<GameStateDTO>("/game/new", {
    method: "POST",
    body: JSON.stringify({}),
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
export function startTrain(total_timesteps?: number) {
  return request<{ status: string }>("/train/start", {
    method: "POST",
    body: JSON.stringify({ total_timesteps: total_timesteps ?? 500000 }),
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
