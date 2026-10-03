/**
 * The table's sound engine (client-only singleton).
 *
 * Every event plays a recorded file from the manifest when it has loaded
 * (plus its synthesized accent, if any), otherwise its synthesized recipe.
 * Spoken lines play a generated voice file or, failing that, the browser's
 * speech synthesis in Portuguese. Result sounds ("stings") never pile up: a
 * match result cuts the hand result, which cuts the round result.
 *
 * The files and the manifest come from tools/sounds/build_sounds.py.
 */

import type { Cue, SoundEvent, Speaker } from "./cues";
import { SAMPLES, VOICE_FILES, VOICE_LINES } from "./manifest";
import { ACCENTS, RECIPES, roomImpulse } from "./synth";

export interface SoundSettings {
  volume: number; // 0..1
  muted: boolean;
  voices: boolean;
}

export const DEFAULT_SOUND_SETTINGS: SoundSettings = { volume: 0.8, muted: false, voices: true };

interface EventMeta {
  /** Reverb send level. */
  wet: number;
  /** Result priority; a sting cuts the active ones of equal or lower priority. */
  sting?: number;
  /** Seconds before the spoken line, so the effect's attack does not mask it. */
  voiceAt?: number;
}

const META: Record<SoundEvent, EventMeta> = {
  shuffle: { wet: 0.1 },
  deal: { wet: 0.1 },
  card: { wet: 0.08 },
  // Bets: the voice comes in after the knocks.
  truco: { wet: 0.25, voiceAt: 0.18 },
  seis: { wet: 0.25, voiceAt: 0.18 },
  nove: { wet: 0.25, voiceAt: 0.28 },
  doze: { wet: 0.3, voiceAt: 0.28 },
  accept: { wet: 0.2, voiceAt: 0.1 },
  run: { wet: 0.2, voiceAt: 0.08 },
  roundWin: { wet: 0.3, sting: 1 },
  roundLose: { wet: 0.3, sting: 1 },
  roundTie: { wet: 0.2, sting: 1 },
  handWin: { wet: 0.3, sting: 2 },
  handLose: { wet: 0.3, sting: 2 },
  handDraw: { wet: 0.2, sting: 2 },
  mao11: { wet: 0.3, sting: 2 },
  matchWin: { wet: 0.35, sting: 3, voiceAt: 1.0 },
  matchLose: { wet: 0.35, sting: 3, voiceAt: 1.2 },
};

/** Effect level while a voice line plays over it. */
const DUCK_GAIN = 0.35;

const STORAGE_KEY = "truco:sound";
const STING_SECONDS = 3;

const pick = <T,>(items: T[]): T => items[Math.floor(Math.random() * items.length)];

function readSettings(): SoundSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT_SOUND_SETTINGS, ...JSON.parse(raw) };
  } catch {
    /* storage blocked or corrupt: defaults */
  }
  return DEFAULT_SOUND_SETTINGS;
}

class SoundEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private reverb: ConvolverNode | null = null;
  private buffers = new Map<string, AudioBuffer | null>();
  private fetched = new Map<string, Promise<ArrayBuffer>>();
  private voice: AudioBufferSourceNode | null = null;
  private stings: { group: GainNode; priority: number; until: number }[] = [];
  private timers = new Set<ReturnType<typeof setTimeout>>();
  private settings: SoundSettings = DEFAULT_SOUND_SETTINGS;
  private listeners = new Set<() => void>();

  constructor() {
    if (typeof window !== "undefined") this.settings = readSettings();
  }

  // --- settings (useSyncExternalStore) -----------------------------------
  getSettings = (): SoundSettings => this.settings;

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  setSettings = (patch: Partial<SoundSettings>): void => {
    const next = { ...this.settings, ...patch };
    next.volume = Math.min(1, Math.max(0, next.volume));
    this.settings = next;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* not persisted */
    }
    if (this.ctx && this.master) {
      this.master.gain.setTargetAtTime(this.level(), this.ctx.currentTime, 0.02);
    }
    if (next.muted || !next.voices) this.stopSpeech();
    this.listeners.forEach((fn) => fn());
  };

  private level(): number {
    return this.settings.muted ? 0 : this.settings.volume;
  }

  // --- playback ------------------------------------------------------------
  /** Download every sound file ahead of time (no audio context needed). */
  prefetch = (): void => {
    if (typeof window === "undefined") return;
    for (const url of allFiles()) {
      if (!this.fetched.has(url)) {
        const p = fetch(url).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(String(r.status)))));
        p.catch(() => {});
        this.fetched.set(url, p);
      }
    }
  };

  /** Create/resume the audio context and decode the files. Call from a user gesture. */
  unlock = (): void => {
    const ctx = this.context();
    if (!ctx) return;
    if (typeof window !== "undefined") window.speechSynthesis?.getVoices(); // starts loading voices
    this.decodeAll(ctx);
  };

  /** Play cues at their offsets (ms). */
  playCues = (cues: Cue[]): void => {
    for (const cue of cues) {
      if (cue.at <= 0) {
        this.play(cue.event, cue.by);
        continue;
      }
      const id = setTimeout(() => {
        this.timers.delete(id);
        this.play(cue.event, cue.by);
      }, cue.at);
      this.timers.add(id);
    }
  };

  play = (event: SoundEvent, by?: Speaker): void => {
    if (this.settings.muted) return;
    const ctx = this.context();
    if (!ctx || !this.master || !this.reverb) return;
    const meta = META[event];
    const now = ctx.currentTime;

    if (meta.sting != null) {
      this.stings = this.stings.filter((s) => s.until > now);
      if (this.stings.some((s) => s.priority > meta.sting!)) return;
      this.stings.forEach((s) => this.fadeOut(s.group, now));
      this.stings = [];
    }

    const group = ctx.createGain();
    group.connect(this.master);
    const send = ctx.createGain();
    send.gain.value = meta.wet;
    group.connect(send).connect(this.reverb);
    if (meta.sting != null) {
      this.stings.push({ group, priority: meta.sting, until: now + STING_SECONDS });
    }

    const sample = this.loadedBuffer(SAMPLES[event]);
    if (sample) {
      const src = ctx.createBufferSource();
      src.buffer = sample;
      src.playbackRate.value = 1 + (Math.random() * 2 - 1) * 0.03;
      src.connect(group);
      src.start(now);
      ACCENTS[event]?.(ctx, group, now + 0.01);
    } else {
      RECIPES[event](ctx, group, now + 0.01);
    }

    if (this.settings.voices) this.speak(event, by ?? "p1", group, now + (meta.voiceAt ?? 0));
  };

  /** Drop scheduled cues and silence results and speech (e.g. on a new match). */
  cancel = (): void => {
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    if (this.ctx) this.stings.forEach((s) => this.fadeOut(s.group, this.ctx!.currentTime));
    this.stings = [];
    this.stopSpeech();
  };

  // --- internals -----------------------------------------------------------
  private context(): AudioContext | null {
    if (typeof window === "undefined") return null;
    if (!this.ctx) {
      const Ctor: typeof AudioContext | undefined =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return null;
      const ctx = new Ctor();
      const compressor = ctx.createDynamicsCompressor();
      compressor.threshold.value = -14;
      compressor.ratio.value = 4;
      compressor.connect(ctx.destination);
      this.master = ctx.createGain();
      this.master.gain.value = this.level();
      this.master.connect(compressor);
      this.reverb = ctx.createConvolver();
      this.reverb.buffer = roomImpulse(ctx);
      this.reverb.connect(this.master);
      this.ctx = ctx;
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  private fadeOut(group: GainNode, now: number) {
    group.gain.cancelScheduledValues(now);
    group.gain.setValueAtTime(group.gain.value, now);
    group.gain.linearRampToValueAtTime(0, now + 0.08);
    setTimeout(() => group.disconnect(), 200);
  }

  private decodeAll(ctx: AudioContext) {
    this.prefetch();
    for (const [url, data] of this.fetched) {
      if (this.buffers.has(url)) continue;
      this.buffers.set(url, null); // missing until decoded; falls back to synthesis
      data
        .then((d) => ctx.decodeAudioData(d.slice(0)))
        .then((buf) => this.buffers.set(url, buf))
        .catch(() => {});
    }
  }

  private loadedBuffer(urls: string[] | undefined): AudioBuffer | null {
    const ready = (urls ?? []).map((u) => this.buffers.get(u)).filter((b): b is AudioBuffer => !!b);
    return ready.length ? pick(ready) : null;
  }

  /** Speak ``event``'s line at ``at`` (context time), ducking ``effect`` under it. */
  private speak(event: SoundEvent, by: Speaker, effect: GainNode, at: number) {
    const file = this.loadedBuffer(VOICE_FILES[event]?.[by]);
    if (file && this.ctx && this.master) {
      this.stopSpeech(); // one line at a time
      const src = this.ctx.createBufferSource();
      src.buffer = file;
      src.connect(this.master);
      src.start(at);
      this.voice = src;
      const g = effect.gain;
      g.setValueAtTime(1, at);
      g.linearRampToValueAtTime(DUCK_GAIN, at + 0.06);
      g.setValueAtTime(DUCK_GAIN, at + Math.max(0.06, file.duration - 0.1));
      g.linearRampToValueAtTime(1, at + file.duration + 0.15);
      return;
    }
    const lines = VOICE_LINES[event]?.[by];
    const synth = typeof window !== "undefined" ? window.speechSynthesis : undefined;
    if (!lines?.length || !synth) return;
    if (this.ctx && at > this.ctx.currentTime + 0.02) {
      const id = setTimeout(() => {
        this.timers.delete(id);
        this.speakText(synth, lines, by);
      }, (at - this.ctx.currentTime) * 1000);
      this.timers.add(id);
      return;
    }
    this.speakText(synth, lines, by);
  }

  private speakText(synth: SpeechSynthesis, lines: string[], by: Speaker) {
    const voice = this.voiceFor(synth, by);
    if (!voice) return; // no Portuguese voice installed: stay quiet
    const u = new SpeechSynthesisUtterance(pick(lines));
    u.voice = voice;
    u.lang = voice.lang;
    u.rate = 1.1;
    u.pitch = by === "p1" ? 0.75 : 1.1;
    u.volume = this.settings.volume;
    synth.cancel();
    synth.speak(u);
  }

  /** The human and the AI get different Portuguese voices when two exist. */
  private voiceFor(synth: SpeechSynthesis, by: Speaker): SpeechSynthesisVoice | null {
    const pt = synth
      .getVoices()
      .filter((v) => v.lang.toLowerCase().replace("_", "-").startsWith("pt"))
      .sort((a, b) => Number(b.lang.toLowerCase().includes("br")) - Number(a.lang.toLowerCase().includes("br")));
    if (!pt.length) return null;
    return by === "p1" ? pt[1] ?? pt[0] : pt[0];
  }

  private stopSpeech() {
    try {
      this.voice?.stop();
    } catch {
      /* already stopped */
    }
    this.voice = null;
    if (typeof window !== "undefined") window.speechSynthesis?.cancel();
  }
}

function allFiles(): string[] {
  return [
    ...Object.values(SAMPLES).flat(),
    ...Object.values(VOICE_FILES).flatMap((bySpeaker) => Object.values(bySpeaker ?? {}).flat()),
  ] as string[];
}

export const sound = new SoundEngine();
