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
  /** Player at 11 in a Mão de 11 hand (they decide to play or run). */
  mao11_player: number | null;
  match_winner: number | null;
  terminated: boolean;
  legal_actions: number[];

  opponent?: { ref: string; label: string };
  stats?: { p0: GameSideStats; p1: GameSideStats };
  vira?: CardDTO;
  p0_hand?: CardDTO[];
  p1_cards_left?: number;
  stake?: number;
  pending_stake?: number | null;
  truco_caller?: number | null;
  current_player?: number;
  awaiting_mao11_response?: boolean;
  rounds?: RoundDTO[];
  hand_winner?: number | null;
  /** Three tied rounds: the hand ended and nobody scored. */
  hand_drawn?: boolean;
  hand_ending?: boolean;
  /** A round just closed and the AI opens the next one; call /game/continue. */
  ai_pending?: boolean;
}

export interface GameSideStats {
  truco_calls: number;
  raises: number;
  accepts: number;
  runs: number;
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
  run_id?: string | null;
  total_timesteps?: number;
  error?: string | null;
}

export type WSMessage =
  | { type: "metrics"; items: EvalMetric[] }
  | ({ type: "status" } & TrainStatus);
