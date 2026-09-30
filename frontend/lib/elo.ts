// Bradley–Terry ratings on the Elo scale from pairwise win counts.
//
// Each result says "a won `wins` of `games` against b". Ratings are fit by
// maximizing the likelihood with a weak prior toward 0 (keeps players with
// only wins or only losses finite), then shifted so `anchor` sits at 0.

export interface PairResult {
  a: string;
  b: string;
  wins: number;
  games: number;
}

const SCALE = 400 / Math.LN10;

export function fitElo(results: PairResult[], anchor?: string): Record<string, number> {
  const players = Array.from(new Set(results.flatMap((r) => [r.a, r.b])));
  const idx = new Map(players.map((p, i) => [p, i]));
  const theta = new Float64Array(players.length);
  const prior = 0.01;
  // Diagonal Newton steps (half-damped): stable for these convex fits.
  for (let iter = 0; iter < 500; iter++) {
    const grad = new Float64Array(players.length);
    const hess = new Float64Array(players.length);
    for (const r of results) {
      const i = idx.get(r.a)!;
      const j = idx.get(r.b)!;
      const p = 1 / (1 + Math.exp(theta[j] - theta[i]));
      const g = r.wins - r.games * p;
      const h = r.games * p * (1 - p);
      grad[i] += g;
      grad[j] -= g;
      hess[i] += h;
      hess[j] += h;
    }
    let maxStep = 0;
    for (let k = 0; k < players.length; k++) {
      const step = (0.5 * (grad[k] - prior * theta[k])) / (hess[k] + prior);
      theta[k] += step;
      maxStep = Math.max(maxStep, Math.abs(step));
    }
    if (maxStep < 1e-6) break;
  }
  const shift = anchor != null && idx.has(anchor) ? theta[idx.get(anchor)!] : 0;
  const out: Record<string, number> = {};
  players.forEach((p, k) => {
    out[p] = (theta[k] - shift) * SCALE;
  });
  return out;
}
