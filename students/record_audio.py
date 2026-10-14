#!/usr/bin/env python3
"""Record your own test clip: N seconds from the microphone -> 16 kHz mono WAV.

Requires: pip install sounddevice   (bundles PortAudio)
Usage:   python3 students/record_audio.py --seconds 10 --out my_voice.wav

Prefer system tools instead? One-liners in README_STUDENTS.md ("From home"):
  macOS:   ffmpeg -f avfoundation -i ":0"  -t 10 -ar 16000 -ac 1 my_voice.wav
  Linux:   ffmpeg -f pulse -i default -t 10 -ar 16000 -ac 1 my_voice.wav
  phone:   ffmpeg -i voice.m4a -ar 16000 -ac 1 my_voice.wav

NOTE: written for student laptops; untested on the lab server (which has no
microphone) — if anything misbehaves, use the ffmpeg one-liners above.
"""
import argparse
import wave

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--out', default='my_voice.wav')
    ap.add_argument('--sr', type=int, default=16000)
    a = ap.parse_args()
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("sounddevice not installed. Run: pip install sounddevice\n"
                         "Or use the ffmpeg one-liners in README_STUDENTS.md.")
    print(f"Recording {a.seconds:.0f} s at {a.sr} Hz mono... speak now!")
    audio = sd.rec(int(a.seconds * a.sr), samplerate=a.sr, channels=1, dtype='float32')
    sd.wait()
    x = np.clip(audio[:, 0], -1.0, 1.0)
    with wave.open(a.out, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(a.sr)
        w.writeframes((x * 32767).astype('<i2').tobytes())
    print(f"saved {a.out} ({len(x)/a.sr:.1f} s, 16 kHz mono) — ready for the students/ scripts")


if __name__ == '__main__':
    main()
