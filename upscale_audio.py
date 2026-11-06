# -*- coding: utf-8 -*-
"""
Chuyen audio 24 kHz -> 48 kHz, co the bat/tat EQ high-shelf.

Vi du:
  python upscale_audio.py                                # ban co EQ +2 dB
  python upscale_audio.py --eq-gain 0 -o out_noeq.wav    # ban khong EQ
  python upscale_audio.py --start 60 --duration 20 -o ab_eq.wav

Khac voi doan code goc:
  - Dung resample_poly (polyphase, up=2/down=1) thay cho FFT resample:
    nhanh hon, khong bi "ringing"/echo o dau-cuoi file.
  - Khong dung highpass 8 kHz (loc do se cat gan het giong noi).
    Thay bang high-shelf nhe de tang do "air".
"""

import argparse

import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 48000


def high_shelf_sos(fs, f0, gain_db, q=0.707):
    """High-shelf biquad theo cong thuc RBJ cookbook."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def process(src, dst, eq_gain_db, eq_freq, eq_q, start=None, duration=None):
    audio, sr = sf.read(src, always_2d=True, dtype="float64")

    # Cat doan trich (neu co) truoc khi resample
    if start is not None or duration is not None:
        i0 = int(round((start or 0.0) * sr))
        i1 = audio.shape[0] if duration is None else int(round((start or 0.0) * sr + duration * sr))
        audio = audio[i0:i1]
        print(f"Trich  : {start or 0.0:.2f}s -> {start or 0.0}+{len(audio) / sr:.2f}s")

    print(f"Input  : {src} | {sr} Hz | {audio.shape[1]} ch | {len(audio) / sr:.2f} s")

    # 1. Resample 24 kHz -> 48 kHz (polyphase)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        up, down = TARGET_SR // g, int(sr) // g
        audio = signal.resample_poly(audio, up, down, axis=0)
        print(f"Resample: {sr} -> {TARGET_SR} Hz (up={up}, down={down})")

    # 2. High-shelf (chi khi gain khac 0)
    if abs(eq_gain_db) > 1e-9:
        sos = high_shelf_sos(TARGET_SR, eq_freq, eq_gain_db, eq_q)
        audio = signal.sosfiltfilt(sos, audio, axis=0)
        print(f"EQ     : high-shelf {eq_gain_db:+.1f} dB @ {eq_freq:.0f} Hz (Q={eq_q})")
    else:
        print("EQ     : khong ap dung")

    # 3. Chong clipping
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio *= 0.999 / peak
        print(f"Peak   : {peak:.3f} -> ha gain xuong 0.999")

    # 4. Ghi file 48 kHz, 24-bit PCM
    sf.write(dst, audio, TARGET_SR, subtype="PCM_24")
    info = sf.info(dst)
    print(f"Output : {dst} | {info.samplerate} Hz | {info.channels} ch | "
          f"{info.duration:.2f} s | {info.subtype}")


def main():
    p = argparse.ArgumentParser(description="Upsample 24 kHz -> 48 kHz + EQ high-shelf")
    p.add_argument("-i", "--input", default="audio.wav")
    p.add_argument("-o", "--output", default="out_high_res.wav")
    p.add_argument("--eq-gain", type=float, default=2.0, help="dB, 0 = tat EQ (default 2.0)")
    p.add_argument("--eq-freq", type=float, default=8000.0, help="Hz (default 8000)")
    p.add_argument("--eq-q", type=float, default=0.707)
    p.add_argument("--start", type=float, default=None, help="giay bat dau trich")
    p.add_argument("--duration", type=float, default=None, help="do dai doan trich (giay)")
    a = p.parse_args()

    process(a.input, a.output, a.eq_gain, a.eq_freq, a.eq_q, a.start, a.duration)


if __name__ == "__main__":
    main()
