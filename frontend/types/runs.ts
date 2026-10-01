// Shapes returned by /runs/* and /arena/* (see backend/api/runs.py and
// backend/api/arena_jobs.py).

export interface EvalResult {
  opponent?: string;
  games: number;
  wins?: number;
  win_rate: number;
  ci95: [number, number];
  reward_mean?: number;
  points_per_hand?: number;
  opp_points_per_hand?: number;
  truco_rate?: number;
  run_rate?: number;
  fallbacks?: number;
}

export interface TrainBlock {
  episodes: number;
  win_rate: number | null;
  reward_mean: number | null;
  by_opponent: Record<string, { games: number; win_rate: number | null }>;
  truco_rate: number | null;
  run_rate: number | null;
  entropy: number | null;
  approx_kl?: number;
  clip_fraction?: number;
  value_loss?: number;
  explained_variance?: number;
}

export interface RunMetric {
  timestep: number;
  episode: number;
  elapsed_s: number;
  checkpoint: string;
  train: TrainBlock;
  eval: Record<string, EvalResult>;
  eval_score: number | null;
  eval_score_ci95: [number, number] | null;
}

export interface RunSummary {
  id: string;
  name: string;
  status: "running" | "finished" | "stopped" | "failed";
  started_at: number;
  ended_at: number | null;
  timesteps: number;
  total_timesteps: number;
  obs_version: string;
  best: { checkpoint: string; score: number; timestep: number } | null;
  error: string | null;
  duration_s: number | null;
  config: {
    obs_version?: string;
    total_timesteps?: number;
    seed?: number;
    league?: Record<string, number>;
    eval_games?: number;
    eval_opponents?: string[];
    hyperparams?: Record<string, unknown>;
  };
  git: { commit: string | null; dirty: boolean | null } | null;
  n_metrics: number;
  last_eval: {
    timestep: number;
    eval_score: number | null;
    win_rates: Record<string, number>;
  } | null;
}

export interface CheckpointInfo {
  ref: string;
  name: string;
  timestep: number | null;
  kind: "best" | "final" | "snapshot";
}

export interface MatrixRow {
  checkpoint: string;
  timestep: number | null;
  results: Record<string, { win_rate: number; ci95: [number, number]; games: number }>;
}

export interface Matrix {
  opponents: string[];
  rows: MatrixRow[];
}

export interface PlayerOption {
  ref: string;
  label: string;
  spec?: string;
}

export interface PlayersList {
  levels: PlayerOption[];
  builtin: string[];
  models: PlayerOption[];
  runs: { run_id: string; name: string; checkpoints: CheckpointInfo[] }[];
  default: string;
}

export interface SideStats {
  decisions: number;
  illegal: number;
  fallbacks: number;
  errors: number;
  truco_calls: number;
  raises: number;
  accepts: number;
  runs: number;
  bet_responses: number;
  points: number;
  hands_won: number;
  points_per_hand: number;
  truco_rate: number;
  run_rate: number;
}

export interface MatchupSummary {
  a: string;
  b: string;
  games: number;
  a_wins: number;
  win_rate_a: number;
  ci95: [number, number];
  hands: number;
  drawn_hands: number;
  mean_hands_per_game: number;
  a_stats: SideStats;
  b_stats: SideStats;
  by_seat: Record<string, { games: number; a_wins: number; win_rate: number }>;
  seed: number;
  elapsed_s: number;
}

export interface ArenaJob {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  a: string;
  b: string;
  a_label: string;
  b_label: string;
  games: number;
  seed: number;
  run_id: string | null;
  progress: { done: number; total: number };
  result: MatchupSummary | null;
  error: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}
