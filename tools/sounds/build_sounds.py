"""Build the table's sound files and their manifest.

    pip install -r tools/sounds/requirements.txt
    python tools/sounds/build_sounds.py

Downloads the sources into tools/sounds/.cache/ (Kenney CC0 packs, Piper
pt_BR voices), mixes the effects, synthesizes the spoken lines and writes:

    frontend/public/sounds/sfx/*.mp3
    frontend/public/sounds/voice/{p0,p1}/*.mp3
    frontend/lib/sound/manifest.ts     (generated: SAMPLES, VOICE_FILES, VOICE_LINES)

Edit SFX or LINES below and run it again. p0 is the human, p1 the AI.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / ".cache"
OUT = ROOT / "frontend" / "public" / "sounds"
MANIFEST = ROOT / "frontend" / "lib" / "sound" / "manifest.ts"
URL_PREFIX = "/sounds"
FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 44100

KENNEY = {
    "casino": "https://kenney.nl/media/pages/assets/casino-audio/2472606a04-1721639069/kenney_casino-audio.zip",
    "impact": "https://kenney.nl/media/pages/assets/impact-sounds/87b4ddecda-1677589768/kenney_impact-sounds.zip",
    "jingles": "https://kenney.nl/media/pages/assets/music-jingles/f37e530b9e-1677590399/kenney_music-jingles.zip",
}
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/{v}/medium/pt_BR-{v}-medium.{ext}"

# --- Effects ------------------------------------------------------------------
# event -> variants; a variant is a list of layers (kenney file stem, delay s,
# gain dB[, ffmpeg filter]). Peak level per event in dBFS.
Layer = Tuple
SFX: Dict[str, List[List[Layer]]] = {
    "card": [[(f"card-place-{i}", 0, 0)] for i in (1, 2, 3, 4)],
    "shuffle": [[("card-shuffle", 0, 0, "atrim=0:1.2,afade=t=out:st=0.85:d=0.35"), ("card-slide-1", "end+0.10", -3),
                 ("card-slide-3", "end+0.24", -3), ("card-slide-5", "end+0.38", -3)]],
    "deal": [[(f"card-slide-{i}", 0.09 * k, -2) for k, i in enumerate(ids)]
             for ids in ((1, 2, 3, 4, 5, 6), (3, 4, 5, 6, 7, 8))],
    "truco": [[("impactWood_medium_000", 0, 0), ("impactWood_medium_001", 0.12, 0)]],
    "seis": [[("impactWood_heavy_000", 0, 0), ("impactWood_heavy_001", 0.12, 0)]],
    "nove": [[("impactWood_heavy_000", 0, 0), ("impactWood_heavy_001", 0.11, 0),
              ("impactWood_heavy_002", 0.22, 0)]],
    "doze": [[("impactWood_heavy_000", 0, 0), ("impactWood_heavy_001", 0.10, 0),
              ("impactWood_heavy_002", 0.20, 0), ("chips-collide-1", 0.32, -2)]],
    "accept": [[(f"chip-lay-{i}", 0, 0)] for i in (1, 2, 3)],
    "run": [[(f"card-shove-{i}", 0, 0)] for i in (1, 2, 3, 4)],
    "roundWin": [[("jingles_PIZZI16", 0, 0)]],
    "roundLose": [[("jingles_PIZZI08", 0, 0)]],
    "roundTie": [[("impactWood_light_000", 0, 0), ("impactWood_light_001", 0.14, 0)]],
    "handWin": [[("chips-collide-2", 0, 0), ("jingles_PIZZI06", 0.05, -2)],
                [("chips-collide-3", 0, 0), ("jingles_PIZZI15", 0.05, -2)]],
    "handLose": [[("jingles_PIZZI05", 0, 0), ("card-slide-2", 0.1, -8)],
                 [("jingles_PIZZI11", 0, 0), ("card-slide-4", 0.1, -8)]],
    "handDraw": [[("impactWood_light_002", 0, 0), ("impactWood_light_003", 0.14, 0)]],
    "mao11": [[("impactSoft_heavy_000", 0, 0, "lowpass=f=500"), ("impactSoft_heavy_001", 0.22, -4, "lowpass=f=500"),
               ("impactSoft_heavy_002", 0.80, 0, "lowpass=f=500"), ("impactSoft_heavy_003", 1.02, -4, "lowpass=f=500")]],
    "matchWin": [[("jingles_STEEL02", 0, 0), ("chips-collide-1", 0.5, -4),
                  ("chips-handle-2", 1.0, -4), ("chips-collide-4", 1.4, -6)]],
    "matchLose": [[("jingles_SAX07", 0, 0)]],
}
PEAK_DB = {"card": -5, "shuffle": -6, "deal": -8, "accept": -6, "run": -6,
           "roundWin": -10, "roundLose": -10, "roundTie": -9, "handDraw": -9,
           "handWin": -5, "handLose": -6, "mao11": -3, "matchWin": -3, "matchLose": -3}
DEFAULT_PEAK_DB = -3

# --- Voices -------------------------------------------------------------------
VOICES = {
    "p0": {"model": "faber", "length_scale": 1.0},
    "p1": {"model": "cadu", "length_scale": 0.88},
}
LINES: Dict[str, Dict[str, List[str]]] = {
    "truco": {"p0": ["Truco!", "Truco, marreco!"], "p1": ["Truco!", "Truco, patinho!", "Truco, ladrão!"]},
    "seis": {"p0": ["Seis!", "Seis, ladrão!"], "p1": ["Seis!", "Seis, malandro!"]},
    "nove": {"p0": ["Nove!", "Nove, que eu tô com tudo!"], "p1": ["Nove!", "Nove, e não chora!"]},
    "doze": {"p0": ["Doze!", "Doze, vamo acabar com isso!"], "p1": ["Doze!", "Doze, pra fechar!"]},
    "accept": {"p0": ["Cai dentro!", "Vem!", "Aceito!"], "p1": ["Cai dentro!", "Vem!", "Bora!"]},
    "run": {"p0": ["Tô fora.", "Corri."], "p1": ["Tô fora.", "Essa eu deixo.", "Fica pra próxima."]},
    "matchWin": {"p1": ["Foi sorte, malandro.", "Hoje não foi meu dia."]},
    "matchLose": {"p1": ["Vai treinar, patinho!", "Volta pro bar, marreco!"]},
}
# A slow attack keeps the first consonant intact; Piper starts speaking at
# sample 0, so each line gets a little silence and a fade-in before it.
VOICE_FILTER = "highpass=f=90,acompressor=threshold=0.1:ratio=3:attack=20:release=120"
VOICE_PEAK_DB = -1.5
VOICE_LEAD_S = 0.06
VOICE_FADE_IN_S = 0.008


# --- Helpers ------------------------------------------------------------------
def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  baixando {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    return dest


def fetch_sources() -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for name, url in KENNEY.items():
        z = download(url, CACHE / f"{name}.zip")
        d = CACHE / name
        if not d.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
        for p in d.rglob("*.ogg"):
            files.setdefault(p.stem, p)
    for cfg in VOICES.values():
        for ext in ("onnx", "onnx.json"):
            v = cfg["model"]
            download(PIPER.format(v=v, ext=ext), CACHE / f"pt_BR-{v}-medium.{ext}")
    return files


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run([FF, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def run_filter(x: np.ndarray, sr: int, af: str) -> np.ndarray:
    raw = subprocess.run([FF, "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                          "-af", af, "-f", "f32le", "-"],
                         input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def trim(x: np.ndarray, sr: int, db: float = -50, onset_db: float = -30) -> np.ndarray:
    """Cut the silent tail (below ``db``) and everything before the onset
    (``onset_db`` under the peak), so a sound starts the moment it plays."""
    peak = float(np.abs(x).max()) if x.size else 0.0
    loud = np.flatnonzero(np.abs(x) > 10 ** (db / 20))
    onset = np.flatnonzero(np.abs(x) > peak * 10 ** (onset_db / 20))
    if not len(loud) or not len(onset):
        return x
    start = max(0, onset[0] - int(0.005 * sr))
    return x[start:loud[-1] + int(0.03 * sr)]


def finish(x: np.ndarray, sr: int, peak_db: float, out: Path, bitrate: str,
           fade_in_s: float = 0.002) -> None:
    """Normalize to a peak, fade both ends, encode MP3."""
    x = x / max(float(np.abs(x).max()), 1e-6) * 10 ** (peak_db / 20)
    fade = min(len(x), int(0.03 * sr))
    x[len(x) - fade:] *= np.linspace(1, 0, fade)
    fade = min(len(x), int(fade_in_s * sr))
    x[:fade] *= np.linspace(0, 1, fade)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-codec:a", "libmp3lame", "-b:a", bitrate, str(out)],
                   input=x.astype(np.float32).tobytes(), check=True)


def mix(layers: List[Layer], files: Dict[str, Path]) -> np.ndarray:
    tracks: List[Tuple[int, np.ndarray]] = []
    last_end = 0.0
    for layer in layers:
        stem, delay, gain = layer[:3]
        af: Optional[str] = layer[3] if len(layer) > 3 else None
        x = decode(files[stem])
        if af:
            x = run_filter(x, SR, af)
        x = trim(x, SR) * 10 ** (gain / 20)
        if isinstance(delay, str):  # "end+0.10": after the first layer ends
            delay = tracks[0][1].size / SR + float(delay.split("+")[1])
        start = int(delay * SR)
        tracks.append((start, x))
        last_end = max(last_end, start + x.size)
    out = np.zeros(int(last_end), dtype=np.float32)
    for start, x in tracks:
        out[start:start + x.size] += x
    return out


def build_sfx(files: Dict[str, Path]) -> Dict[str, List[str]]:
    samples: Dict[str, List[str]] = {}
    for event, variants in SFX.items():
        urls = []
        for i, layers in enumerate(variants, 1):
            name = f"{event}-{i}.mp3" if len(variants) > 1 else f"{event}.mp3"
            finish(mix(layers, files), SR, PEAK_DB.get(event, DEFAULT_PEAK_DB), OUT / "sfx" / name, "96k")
            urls.append(f"{URL_PREFIX}/sfx/{name}")
        samples[event] = urls
        print(f"  sfx {event}: {len(urls)}")
    return samples


def build_voices() -> Dict[str, Dict[str, List[str]]]:
    from piper import PiperVoice, SynthesisConfig

    loaded = {sp: PiperVoice.load(CACHE / f"pt_BR-{cfg['model']}-medium.onnx") for sp, cfg in VOICES.items()}
    files: Dict[str, Dict[str, List[str]]] = {}
    for event, by_speaker in LINES.items():
        for speaker, lines in by_speaker.items():
            voice = loaded[speaker]
            config = SynthesisConfig(length_scale=VOICES[speaker]["length_scale"], noise_scale=0.75, noise_w_scale=0.9)
            for i, text in enumerate(lines, 1):
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    voice.synthesize_wav(text, w, syn_config=config)
                buf.seek(0)
                with wave.open(buf) as w:
                    sr = w.getframerate()
                    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768
                # Only real silence goes: quiet consonants at the start stay.
                x = trim(x.astype(np.float32), sr, db=-60, onset_db=-60)
                fade = min(len(x), int(VOICE_FADE_IN_S * sr))
                x[:fade] *= np.linspace(0, 1, fade)
                pad = np.zeros(int(VOICE_LEAD_S * sr), np.float32)
                x = run_filter(np.concatenate([pad, x]), sr, VOICE_FILTER)
                name = f"{event}-{i}.mp3"
                finish(x, sr, VOICE_PEAK_DB, OUT / "voice" / speaker / name, "64k", fade_in_s=0)
                files.setdefault(event, {}).setdefault(speaker, []).append(f"{URL_PREFIX}/voice/{speaker}/{name}")
        print(f"  voz {event}")
    return files


def write_manifest(samples, voice_files) -> None:
    def ts(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    MANIFEST.write_text(
        "// Generated by tools/sounds/build_sounds.py; edit that script, not this file.\n"
        'import type { SoundEvent, Speaker } from "./cues";\n\n'
        "type BySpeaker = Partial<Record<Speaker, string[]>>;\n\n"
        "/** Recorded effect files per event; one is picked at random. */\n"
        f"export const SAMPLES: Partial<Record<SoundEvent, string[]>> = {ts(samples)};\n\n"
        "/** Generated voice files per event and speaker. */\n"
        f"export const VOICE_FILES: Partial<Record<SoundEvent, BySpeaker>> = {ts(voice_files)};\n\n"
        "/** The same lines as text, for the browser's speech synthesis when a file is missing. */\n"
        f"export const VOICE_LINES: Partial<Record<SoundEvent, BySpeaker>> = {ts(LINES)};\n",
        encoding="utf-8",
    )


def main() -> None:
    print("Fontes")
    files = fetch_sources()
    if OUT.exists():
        shutil.rmtree(OUT / "sfx", ignore_errors=True)
        shutil.rmtree(OUT / "voice", ignore_errors=True)
    print("Efeitos")
    samples = build_sfx(files)
    print("Vozes")
    voice_files = build_voices()
    write_manifest(samples, voice_files)
    size = sum(p.stat().st_size for p in OUT.rglob("*.mp3"))
    print(f"-> {OUT} ({size / 1024:.0f} KB), {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
