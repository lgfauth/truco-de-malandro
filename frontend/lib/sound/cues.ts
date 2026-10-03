import type { GameStateDTO } from "@/types/game";

/** Every sound the table can make. */
export type SoundEvent =
  | "shuffle"
  | "deal"
  | "card"
  | "truco"
  | "seis"
  | "nove"
  | "doze"
  | "accept"
  | "run"
  | "roundWin"
  | "roundLose"
  | "roundTie"
  | "handWin"
  | "handLose"
  | "handDraw"
  | "mao11"
  | "matchWin"
  | "matchLose";

/** Who makes the sound; picks the voice for spoken lines. */
export type Speaker = "p0" | "p1";

export interface Cue {
  event: SoundEvent;
  /** Delay in ms, relative to the state update that produced the cue. */
  at: number;
  by?: Speaker;
}

const STAKE_EVENT: Record<number, SoundEvent> = { 3: "truco", 6: "seis", 9: "nove", 12: "doze" };

export function stakeEvent(stake: number | null | undefined): SoundEvent | null {
  return stake != null ? STAKE_EVENT[stake] ?? null : null;
}

// Spacing between consecutive AI actions and before the results they cause.
const STEP_MS = 450;
const RESULT_MS = 350;

/**
 * Sounds for the change prev -> cur.
 *
 * The human's own actions are not included: they sound on click, before the
 * round-trip. Results are exclusive: the match result replaces the hand
 * result, which replaces the round result.
 */
export function detectCues(prev: GameStateDTO | null, cur: GameStateDTO): Cue[] {
  if (!prev || prev.game_id !== cur.game_id) return [{ event: "shuffle", at: 0 }];

  const cues: Cue[] = [];
  let t = 0;
  const ai = (event: SoundEvent) => {
    cues.push({ event, at: t, by: "p1" });
    t += STEP_MS;
  };

  // A new hand was dealt: the previous state's rounds belong to the old hand.
  const newHand = !!prev.hand_ending && !cur.hand_ending;
  if (newHand) {
    cues.push({ event: "deal", at: 0 });
    t += STEP_MS;
    if (cur.mao11_player != null || cur.iron_hand) {
      cues.push({ event: "mao11", at: t });
      t += STEP_MS * 2;
    }
  }

  // AI bets: a new or raised pending stake called by the AI.
  const pending = cur.pending_stake ?? null;
  if (pending != null && cur.truco_caller === 1 &&
      (pending !== (prev.pending_stake ?? null) || prev.truco_caller !== 1)) {
    const ev = stakeEvent(pending);
    if (ev) ai(ev);
  }
  // AI accepted the human's bet. Accepting clears truco_caller, so tell it
  // from the human accepting by the value: above what was pending before.
  if (pending == null && !newHand && !prev.awaiting_mao11_response &&
      (cur.stake ?? 1) > (prev.pending_stake ?? prev.stake ?? 1)) {
    ai("accept");
  }

  // AI cards.
  const prevRounds = newHand ? [] : prev.rounds ?? [];
  const curRounds = cur.rounds ?? [];
  curRounds.forEach((round, i) => {
    round.plays.slice(prevRounds[i]?.plays.length ?? 0).forEach((play) => {
      if (play.player === 1) ai("card");
    });
  });

  // AI ran from a bet (or from its Mão de 11).
  if (!newHand && runner(prev, cur) === 1) ai("run");

  // Results, most important only.
  const at = t > 0 ? t - STEP_MS + RESULT_MS : RESULT_MS;
  if (cur.terminated && !prev.terminated) {
    cues.push({ event: cur.match_winner === 0 ? "matchWin" : "matchLose", at });
  } else if (cur.hand_ending && !prev.hand_ending) {
    const event = cur.hand_drawn ? "handDraw" : cur.hand_winner === 0 ? "handWin" : "handLose";
    cues.push({ event, at });
  } else {
    const closed = curRounds.findLast((r, i) => r.result !== null && (prevRounds[i]?.result ?? null) === null);
    if (closed) {
      const event = closed.result === 0 ? "roundWin" : closed.result === 1 ? "roundLose" : "roundTie";
      cues.push({ event, at });
    }
  }
  return cues;
}

/** Who ran in the hand that just ended, or null if nobody did. */
function runner(prev: GameStateDTO, cur: GameStateDTO): number | null {
  const scored = cur.scores.p0 > prev.scores.p0 || cur.scores.p1 > prev.scores.p1;
  if (!scored) return null;
  // A refused bet ends the hand with the bet still pending (frozen hand).
  if (cur.hand_ending && cur.pending_stake != null && cur.truco_caller != null) {
    return cur.truco_caller === 0 ? 1 : 0;
  }
  if (prev.pending_stake != null && prev.truco_caller != null) {
    return prev.truco_caller === 0 ? 1 : 0;
  }
  if (prev.awaiting_mao11_response && prev.mao11_player != null) return prev.mao11_player;
  return null;
}
