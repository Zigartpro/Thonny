"""Fase 1.4: prueba el parlante (MAX98357) por el cable USB.

  1. Un tono de 440 Hz generado en la placa (si no suena: revisar amplificador y parlante).
  2. Una respuesta de voz (por defecto voces/johan/prender_bombillo.wav, "Listo, prendí la luz").
El audio se manda primero a la RAM de la placa (no es tiempo real) y luego suena.

Uso:  python pc/prueba_parlante.py [--wav voces/rocko/que_necesitas.wav] [--volumen 0.8] [--solo-tono]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ  # noqa: E402
from esp32 import Cable  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", default="voces/johan/prender_bombillo.wav", help="WAV de 16 kHz mono (máx. 4 s)")
    ap.add_argument("--volumen", type=float, default=0.8, help="0-1, multiplica la señal")
    ap.add_argument("--solo-tono", action="store_true")
    args = ap.parse_args()

    print("Conectando con la ESP32 por cable…")
    placa = Cable()
    try:
        print("1) Tono de 440 Hz durante 1 s… (debería sonar un pitido)")
        placa.tono(440, 1000, min(args.volumen, 0.3))
        if args.solo_tono:
            return
        wav = Path(args.wav)
        if not wav.is_absolute():
            wav = RAIZ / wav
        x, sr = sf.read(wav, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == 16000, f"{wav.name} está a {sr} Hz; debe ser 16 kHz"
        x = x[: 4 * sr]
        pcm = np.clip(x * args.volumen * 32767, -32768, 32767).astype(np.int16)
        print(f"2) Enviando «{wav.relative_to(RAIZ) if wav.is_relative_to(RAIZ) else wav}» "
              f"({len(pcm) / sr:.1f} s) a la placa…")
        t0 = time.perf_counter()
        placa.reproducir_cable(pcm, al_terminar_envio=lambda: print(
            f"   enviado en {time.perf_counter() - t0:.1f} s. >>> SONANDO <<<"))
        print("   Listo. ¿Se oyó la voz clara?")
    finally:
        placa.cerrar()


if __name__ == "__main__":
    main()
