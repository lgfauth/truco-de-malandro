export const ACTION_PLAY_0 = 0;
export const ACTION_PLAY_1 = 1;
export const ACTION_PLAY_2 = 2;
export const ACTION_CALL_TRUCO = 3;
export const ACTION_ACCEPT = 4;
export const ACTION_RUN = 5;
export const ACTION_RAISE = 6;

export interface CardDTO {
  rank: number;
  suit: number;
  label: string;
}

export interface RoundPlay {
  player: number;
  card: CardDTO;
}

export interface RoundDTO {
  plays: RoundPlay[];
  result: number | null;
}

export interface GameStateDTO {
  game_id: string;
  scores: { p0: number; p1: number };
  dealer: number;
  iron_hand: boolean;
  open_hand_for: number | null;
  match_winner: number | null;
  terminated: boolean;
  legal_actions: number[];

  vira?: CardDTO;
  p0_hand?: CardDTO[];
  p1_hand?: CardDTO[];
  cheat_active?: boolean;
  stake?: number;
  pending_stake?: number | null;
  truco_caller?: number | null;
  current_player?: number;
  awaiting_mao11_response?: boolean;
  rounds?: RoundDTO[];
  hand_winner?: number | null;
  hand_ending?: boolean;
}

export interface EvalMetric {
  timestep: number;
  episode: number;
  win_rate: number;
  mean_reward: number;
  mean_steps: number;
  entropy: number;
  truco_rate: number;
  run_rate: number;
}

export interface TrainStatus {
  running: boolean;
  paused: boolean;
  timestep: number;
  episode: number;
  latest_metric: EvalMetric | null;
}

export type WSMessage =
  | { type: "metrics"; items: EvalMetric[] }
  | ({ type: "status" } & TrainStatus);
