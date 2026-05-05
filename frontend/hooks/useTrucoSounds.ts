"use client";

import { useCallback, useRef } from "react";

type ToneOpts = {
  type?: OscillatorType;
  gain?: number;
  attack?: number;
};

export function useTrucoSounds() {
  const ctxRef = useRef<AudioContext | null>(null);

  const getCtx = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    if (!ctxRef.current) {
      const Ctor: typeof AudioContext | undefined =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) return null;
      ctxRef.current = new Ctor();
    }
    if (ctxRef.current.state === "suspended") {
      void ctxRef.current.resume();
    }
    return ctxRef.current;
  }, []);

  const tone = useCallback(
    (freq: number, duration: number, opts: ToneOpts = {}) => {
      const ctx = getCtx();
      if (!ctx) return;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = opts.type ?? "sawtooth";
      osc.frequency.value = freq;
      const peak = opts.gain ?? 0.25;
      const attack = opts.attack ?? 0.01;
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(peak, now + attack);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now);
      osc.stop(now + duration + 0.02);
    },
    [getCtx]
  );

  const sweep = useCallback(
    (from: number, to: number, duration: number) => {
      const ctx = getCtx();
      if (!ctx) return;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(from, now);
      osc.frequency.exponentialRampToValueAtTime(
        Math.max(to, 1),
        now + duration
      );
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(0.25, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now);
      osc.stop(now + duration + 0.02);
    },
    [getCtx]
  );

  const sequence = useCallback(
    (freqs: number[], step = 0.13) => {
      const ctx = getCtx();
      if (!ctx) return;
      const start = ctx.currentTime;
      freqs.forEach((f, i) => {
        const at = start + i * step;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "triangle";
        osc.frequency.value = f;
        gain.gain.setValueAtTime(0, at);
        gain.gain.linearRampToValueAtTime(0.3, at + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, at + step);
        osc.connect(gain).connect(ctx.destination);
        osc.start(at);
        osc.stop(at + step + 0.02);
      });
    },
    [getCtx]
  );

  const noiseClick = useCallback(
    (duration: number) => {
      const ctx = getCtx();
      if (!ctx) return;
      const now = ctx.currentTime;
      const len = Math.max(1, Math.floor(ctx.sampleRate * duration));
      const buffer = ctx.createBuffer(1, len, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      const filter = ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.value = 2000;
      filter.Q.value = 1;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.4, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
      src.connect(filter).connect(gain).connect(ctx.destination);
      src.start(now);
      src.stop(now + duration);
    },
    [getCtx]
  );

  return {
    playTruco: useCallback(
      () => tone(180, 0.4, { type: "sawtooth", gain: 0.35, attack: 0.005 }),
      [tone]
    ),
    playSeis: useCallback(
      () => tone(220, 0.4, { type: "sawtooth", gain: 0.35, attack: 0.005 }),
      [tone]
    ),
    playNove: useCallback(
      () => tone(260, 0.4, { type: "sawtooth", gain: 0.35, attack: 0.005 }),
      [tone]
    ),
    playDoze: useCallback(
      () => tone(300, 0.4, { type: "sawtooth", gain: 0.35, attack: 0.005 }),
      [tone]
    ),
    playAceitar: useCallback(
      () => tone(440, 0.15, { type: "triangle", gain: 0.3 }),
      [tone]
    ),
    playCorrer: useCallback(() => sweep(300, 150, 0.3), [sweep]),
    playCartaJogada: useCallback(() => noiseClick(0.08), [noiseClick]),
    playVitoria: useCallback(
      () => sequence([440, 550, 660], 0.13),
      [sequence]
    ),
    playDerrota: useCallback(
      () => sequence([440, 370, 300], 0.15),
      [sequence]
    ),
    playRoundWin: useCallback(() => sweep(440, 520, 0.3), [sweep]),
    playRoundLose: useCallback(() => sweep(440, 360, 0.3), [sweep]),
    playRoundTie: useCallback(
      () => sequence([440, 440], 0.1),
      [sequence]
    ),
  };
}
