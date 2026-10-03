"use client";

import { useSoundSettings } from "@/hooks/useSoundSettings";
import { sound } from "@/lib/sound/engine";

/** Mute toggle for the table's top bar. */
export function SoundToggle() {
  const [{ muted, volume }, set] = useSoundSettings();
  const silent = muted || volume === 0;
  return (
    <button
      onClick={() => {
        sound.unlock();
        set(silent ? { muted: false, volume: volume || 0.8 } : { muted: true });
      }}
      className="h-10 w-10 shrink-0 grid place-items-center rounded-xl bg-black/25 hover:bg-black/40 text-lg"
      aria-label={silent ? "Ativar sons" : "Silenciar sons"}
      aria-pressed={silent}
      title={silent ? "Ativar sons" : "Silenciar sons"}
    >
      {silent ? "🔇" : volume < 0.4 ? "🔈" : "🔊"}
    </button>
  );
}

/** Volume, mute and voices, for the game menu. */
export function SoundPanel() {
  const [{ muted, volume, voices }, set] = useSoundSettings();
  return (
    <div className="space-y-2">
      <div className="text-xs uppercase tracking-wide text-zinc-500">Som</div>
      <label className="flex items-center gap-3 text-sm">
        <span className="w-16 text-zinc-400">Volume</span>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={Math.round(volume * 100)}
          onChange={(e) => set({ volume: Number(e.target.value) / 100, muted: false })}
          className="flex-1 accent-amber-500"
          aria-label="Volume"
        />
        <span className="w-9 text-right tabular-nums text-zinc-400">{muted ? "—" : `${Math.round(volume * 100)}%`}</span>
      </label>
      <label className="flex items-center gap-2 text-sm text-zinc-300">
        <input type="checkbox" checked={!muted} onChange={(e) => set({ muted: !e.target.checked })} />
        Som ligado
      </label>
      <label className="flex items-center gap-2 text-sm text-zinc-300">
        <input type="checkbox" checked={voices} onChange={(e) => set({ voices: e.target.checked })} />
        Vozes (“Truco!”, “Cai dentro!”…)
      </label>
      <button
        onClick={() => {
          sound.unlock();
          sound.play("truco", "p1");
        }}
        disabled={muted}
        className="w-full py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-sm font-semibold"
      >
        Testar som
      </button>
    </div>
  );
}
