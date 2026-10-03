"use client";

import { useSyncExternalStore } from "react";

import { DEFAULT_SOUND_SETTINGS, sound, type SoundSettings } from "@/lib/sound/engine";

/** Current sound settings (persisted per browser) and a setter. */
export function useSoundSettings(): [SoundSettings, (patch: Partial<SoundSettings>) => void] {
  const settings = useSyncExternalStore(sound.subscribe, sound.getSettings, () => DEFAULT_SOUND_SETTINGS);
  return [settings, sound.setSettings];
}
