/**
 * Synthesized sounds: layered noise, filtered oscillators and chords.
 *
 * RECIPES is the full sound of each event, used when its recorded file is
 * missing. ACCENTS is only the musical part, played on top of the recorded
 * file (the brass on the bets, the drone under the Mão de 11 heartbeat).
 *
 * Every recipe writes into ``out`` starting at ``t`` (AudioContext time).
 */

import type { SoundEvent } from "./cues";

type Recipe = (ctx: AudioContext, out: AudioNode, t: number) => void;

const noiseCache = new WeakMap<BaseAudioContext, AudioBuffer>();

function noiseBuffer(ctx: BaseAudioContext): AudioBuffer {
  let buf = noiseCache.get(ctx);
  if (!buf) {
    buf = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    noiseCache.set(ctx, buf);
  }
  return buf;
}

const jitter = (v: number, amount = 0.08) => v * (1 + (Math.random() * 2 - 1) * amount);

/** Gain envelope: fast attack, exponential decay. */
function envelope(ctx: AudioContext, t: number, peak: number, attack: number, dur: number): GainNode {
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(Math.max(peak, 0.0002), t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  return g;
}

interface NoiseOpts {
  dur: number;
  gain: number;
  type?: BiquadFilterType;
  freq: number;
  freqEnd?: number;
  q?: number;
  attack?: number;
}

function noise(ctx: AudioContext, out: AudioNode, t: number, o: NoiseOpts) {
  const src = ctx.createBufferSource();
  src.buffer = noiseBuffer(ctx);
  const filter = ctx.createBiquadFilter();
  filter.type = o.type ?? "bandpass";
  filter.Q.value = o.q ?? 1;
  filter.frequency.setValueAtTime(o.freq, t);
  if (o.freqEnd) filter.frequency.exponentialRampToValueAtTime(o.freqEnd, t + o.dur);
  const g = envelope(ctx, t, o.gain, o.attack ?? 0.003, o.dur);
  src.connect(filter).connect(g).connect(out);
  src.start(t, Math.random() * 0.5);
  src.stop(t + o.dur + 0.05);
}

interface ToneOpts {
  freq: number;
  freqEnd?: number;
  dur: number;
  gain: number;
  type?: OscillatorType;
  attack?: number;
  /** Lowpass with an opening/closing envelope (brass-like). */
  filter?: { from: number; peak: number; to: number };
  vibrato?: { rate: number; depth: number };
  detune?: number;
}

function tone(ctx: AudioContext, out: AudioNode, t: number, o: ToneOpts) {
  const osc = ctx.createOscillator();
  osc.type = o.type ?? "sine";
  osc.frequency.setValueAtTime(o.freq, t);
  if (o.freqEnd) osc.frequency.exponentialRampToValueAtTime(o.freqEnd, t + o.dur);
  if (o.detune) osc.detune.value = o.detune;
  let node: AudioNode = osc;
  if (o.filter) {
    const f = ctx.createBiquadFilter();
    f.type = "lowpass";
    f.Q.value = 2;
    f.frequency.setValueAtTime(o.filter.from, t);
    f.frequency.exponentialRampToValueAtTime(o.filter.peak, t + Math.min(0.08, o.dur / 3));
    f.frequency.exponentialRampToValueAtTime(o.filter.to, t + o.dur);
    node = node.connect(f);
  }
  if (o.vibrato) {
    const lfo = ctx.createOscillator();
    const depth = ctx.createGain();
    lfo.frequency.value = o.vibrato.rate;
    depth.gain.value = o.vibrato.depth;
    lfo.connect(depth).connect(osc.frequency);
    lfo.start(t);
    lfo.stop(t + o.dur + 0.05);
  }
  const g = envelope(ctx, t, o.gain, o.attack ?? 0.01, o.dur);
  node.connect(g).connect(out);
  osc.start(t);
  osc.stop(t + o.dur + 0.05);
}

/** Low body hit: a sine dropping in pitch. */
function thump(ctx: AudioContext, out: AudioNode, t: number, freq: number, gain: number, dur = 0.18) {
  tone(ctx, out, t, { freq: freq * 1.8, freqEnd: freq, dur, gain, attack: 0.002 });
}

/** Plucked note: sine body plus a quieter octave partial. */
function pluck(ctx: AudioContext, out: AudioNode, t: number, freq: number, gain: number, dur = 0.6) {
  tone(ctx, out, t, { freq, dur, gain, type: "triangle", attack: 0.004 });
  tone(ctx, out, t, { freq: freq * 2, dur: dur * 0.5, gain: gain * 0.3, attack: 0.002 });
}

/** Small bell: inharmonic partials, as in a coin or a glass. */
function bell(ctx: AudioContext, out: AudioNode, t: number, freq: number, gain: number, dur = 0.5) {
  [1, 2.76, 5.4].forEach((ratio, i) =>
    tone(ctx, out, t, { freq: freq * ratio, dur: dur / (i + 1), gain: gain / (i + 1.5), attack: 0.001 }));
}

/** Brass stab: detuned saws through an opening lowpass. */
function brass(ctx: AudioContext, out: AudioNode, t: number, freq: number, gain: number, dur = 0.45, vibrato = false) {
  for (const detune of [-8, 8]) {
    tone(ctx, out, t, {
      freq, dur, gain: gain / 2, type: "sawtooth", attack: 0.03, detune,
      filter: { from: 300, peak: 2600, to: 700 },
      vibrato: vibrato ? { rate: 5.5, depth: freq * 0.012 } : undefined,
    });
  }
}

function cardSlap(ctx: AudioContext, out: AudioNode, t: number, gain = 1) {
  // Short swish through the air, then the snap on the felt.
  noise(ctx, out, t, { dur: 0.07, gain: 0.12 * gain, freq: jitter(1500), freqEnd: jitter(4500), q: 0.8, attack: 0.02 });
  noise(ctx, out, t + 0.06, { dur: 0.05, gain: 0.55 * gain, type: "highpass", freq: jitter(1800), q: 0.7 });
  thump(ctx, out, t + 0.06, jitter(150), 0.35 * gain, 0.07);
}

function cardSlide(ctx: AudioContext, out: AudioNode, t: number, gain = 1) {
  noise(ctx, out, t, { dur: 0.11, gain: 0.25 * gain, freq: jitter(2200), freqEnd: jitter(5200), q: 1.2, attack: 0.03 });
}

function knock(ctx: AudioContext, out: AudioNode, t: number, gain = 1) {
  thump(ctx, out, t, jitter(95, 0.05), 0.7 * gain, 0.16);
  noise(ctx, out, t, { dur: 0.05, gain: 0.3 * gain, type: "lowpass", freq: 900 });
}

function coins(ctx: AudioContext, out: AudioNode, t: number, count: number, gain = 0.12) {
  for (let i = 0; i < count; i++) {
    bell(ctx, out, t + i * jitter(0.06, 0.4), jitter(2600, 0.25), gain, 0.25);
  }
}

// Note frequencies.
const N = {
  E2: 82.41, A2: 110, C3: 130.81, D3: 146.83, E3: 164.81, F3: 174.61, G3: 196,
  C4: 261.63, E4: 329.63, F4: 349.23, G4: 392, A4: 440, C5: 523.25, E5: 659.25,
  G5: 783.99, Ab5: 830.61, B5: 987.77, C6: 1046.5,
};

/** Knocks per bet level (truco, seis, nove, doze), as in the recorded files. */
const BET_KNOCKS = [2, 2, 3, 3];

/** Brass hit after the knocks: higher and fuller at each level. */
function betBrass(level: number): Recipe {
  const roots = [N.A2, N.C3, N.D3, N.E3];
  return (ctx, out, t) => {
    const at = t + BET_KNOCKS[level] * 0.11;
    const root = roots[level];
    brass(ctx, out, at, root, 0.35 + level * 0.05, 0.5 + level * 0.08);
    brass(ctx, out, at, root * 1.5, 0.2 + level * 0.04, 0.5 + level * 0.08);
    if (level >= 2) brass(ctx, out, at, root * 2, 0.15, 0.6);
    if (level === 3) noise(ctx, out, at, { dur: 1.2, gain: 0.12, type: "highpass", freq: 5000 });
  };
}

/** Bet levels: more knocks and a higher, fuller brass hit each time. */
function bet(level: number): Recipe {
  return (ctx, out, t) => {
    for (let i = 0; i < BET_KNOCKS[level]; i++) knock(ctx, out, t + i * 0.11, 0.8 + level * 0.1);
    betBrass(level)(ctx, out, t);
  };
}

const maoDrone: Recipe = (ctx, out, t) =>
  tone(ctx, out, t, { freq: N.E2, dur: 1.6, gain: 0.12, type: "sawtooth", attack: 0.6,
    filter: { from: 120, peak: 400, to: 150 } });

export const RECIPES: Record<SoundEvent, Recipe> = {
  shuffle: (ctx, out, t) => {
    // Riffle: a burst of tiny clicks speeding up, then the bridge.
    let at = t;
    for (let i = 0; i < 18; i++) {
      noise(ctx, out, at, { dur: 0.02, gain: 0.18, freq: jitter(3500, 0.2), q: 2 });
      at += 0.035 - i * 0.0012;
    }
    noise(ctx, out, at + 0.05, { dur: 0.25, gain: 0.15, freq: 2500, freqEnd: 1200, q: 0.8, attack: 0.05 });
    for (let i = 0; i < 3; i++) cardSlide(ctx, out, at + 0.4 + i * 0.13);
  },
  deal: (ctx, out, t) => {
    for (let i = 0; i < 6; i++) cardSlide(ctx, out, t + i * 0.09, 0.8);
  },
  card: (ctx, out, t) => cardSlap(ctx, out, t),
  truco: bet(0),
  seis: bet(1),
  nove: bet(2),
  doze: bet(3),
  accept: (ctx, out, t) => {
    knock(ctx, out, t, 0.6);
    pluck(ctx, out, t + 0.08, N.C5, 0.18, 0.4);
    pluck(ctx, out, t + 0.16, N.G5, 0.18, 0.6);
  },
  run: (ctx, out, t) => {
    noise(ctx, out, t, { dur: 0.4, gain: 0.25, freq: 2400, freqEnd: 300, q: 1.5, attack: 0.02 });
    pluck(ctx, out, t + 0.1, N.G4, 0.15, 0.3);
    pluck(ctx, out, t + 0.25, N.D3 * 2, 0.15, 0.5);
  },
  roundWin: (ctx, out, t) => {
    pluck(ctx, out, t, N.E5, 0.16, 0.5);
    pluck(ctx, out, t + 0.09, N.Ab5, 0.16, 0.7);
  },
  roundLose: (ctx, out, t) => {
    pluck(ctx, out, t, N.A4 / 2, 0.2, 0.4);
    pluck(ctx, out, t + 0.12, N.F3, 0.2, 0.6);
  },
  roundTie: (ctx, out, t) => {
    for (const d of [0, 0.14]) {
      noise(ctx, out, t + d, { dur: 0.06, gain: 0.25, freq: 1300, q: 6 });
      tone(ctx, out, t + d, { freq: 780, dur: 0.08, gain: 0.12, attack: 0.001 });
    }
  },
  handWin: (ctx, out, t) => {
    coins(ctx, out, t, 7);
    [N.C5, N.E5, N.G5, N.C6].forEach((f, i) => pluck(ctx, out, t + 0.15 + i * 0.06, f, 0.14, 0.9));
  },
  handLose: (ctx, out, t) => {
    [N.A2 * 2, N.C4, N.E4].forEach((f, i) => pluck(ctx, out, t + i * 0.07, f, 0.15, 0.9));
    noise(ctx, out, t + 0.2, { dur: 0.35, gain: 0.12, freq: 3000, freqEnd: 800, q: 1, attack: 0.05 });
  },
  handDraw: (ctx, out, t) => RECIPES.roundTie(ctx, out, t),
  mao11: (ctx, out, t) => {
    // Heartbeat over a low swelling drone.
    for (const [d, gain] of [[0, 0.8], [0.22, 0.55], [0.8, 0.8], [1.02, 0.55]]) {
      thump(ctx, out, t + d, 55, gain, 0.25);
    }
    maoDrone(ctx, out, t);
  },
  matchWin: (ctx, out, t) => {
    [N.C4, N.E4, N.G4].forEach((f, i) => brass(ctx, out, t + i * 0.14, f, 0.3, 0.2));
    const at = t + 0.45;
    [N.C4, N.E4, N.G4, N.C5].forEach((f) => brass(ctx, out, at, f, 0.22, 1.6, true));
    noise(ctx, out, at, { dur: 1.8, gain: 0.1, type: "highpass", freq: 6000 });
    coins(ctx, out, at + 0.2, 10, 0.1);
  },
  matchLose: (ctx, out, t) => {
    // Sad trombone.
    [N.G3, 185, N.F3].forEach((f, i) => brass(ctx, out, t + i * 0.38, f, 0.32, 0.36));
    brass(ctx, out, t + 3 * 0.38, N.E3, 0.32, 1.3, true);
  },
};

export const ACCENTS: Partial<Record<SoundEvent, Recipe>> = {
  truco: betBrass(0),
  seis: betBrass(1),
  nove: betBrass(2),
  doze: betBrass(3),
  mao11: maoDrone,
};

/** A short decaying stereo noise impulse: a small room for the reverb. */
export function roomImpulse(ctx: BaseAudioContext, seconds = 1.4): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
  }
  return buf;
}
