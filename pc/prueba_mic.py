"""Fase 1.4: graba unos segundos del INMP441 (vía la ESP32) y los guarda en un WAV.

Muestra el nivel (RMS y pico en dBFS) cada medio segundo y al final, para saber si el
micrófono suena bien, muy bajo (subir ganancia: bajar MIC_SHIFT en config.h) o saturado.

Uso:  python pc/prueba_mic.py [--segundos 5] [--host 192.168.x.x | --cable [COM12]]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, ruta  # noqa: E402
from esp32 import Cable, Microfono  # noqa: E402


def dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean((x / 32768.0) ** 2)) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segundos", type=float, default=5)
    ap.add_argument("--host", help="IP de la ESP32 (por defecto la de config.yaml)")
    ap.add_argument("--cable", nargs="?", const="", metavar="COMx",
                    help="usar el cable USB en vez de WiFi (puerto de config.yaml si no se da)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.cable is not None and config()["esp32"]["firmware"] == "micropython":
        # MicroPython por cable: la placa graba en su RAM y luego envía (no es tiempo real)
        print("Conectando con la ESP32 por cable…")
        mic = Cable(args.cable or None)
        sr = mic.sr
        segundos = min(args.segundos, 3)
        print(f"Grabando {segundos} s: HABLA AHORA cerca del micrófono.")
        audio = mic.grabar(segundos)
        print("Recibido. Nivel cada medio segundo:")
        trozos = [audio[i:i + sr // 2] for i in range(0, len(audio), sr // 2)]
        for x in trozos:
            nivel = dbfs(x)
            print(f"  {nivel:6.1f} dBFS |{'#' * max(0, int((nivel + 70) / 2))}")
        mic.cerrar()
    else:
        if args.cable is not None:
            print("Conectando con la ESP32 por cable…")
            mic = Cable(args.cable or None)
            mic.modo_audio()
        else:
            print("Conectando con la ESP32 por WiFi…")
            mic = Microfono(args.host)
        sr = mic.sr
        print(f"Grabando {args.segundos} s: habla cerca del micrófono.")
        trozos = []
        for _ in range(int(args.segundos * 2)):
            x = mic.leer(sr // 2)
            trozos.append(x)
            nivel = dbfs(x)
            print(f"  {nivel:6.1f} dBFS |{'#' * max(0, int((nivel + 70) / 2))}")
        mic.cerrar()
        audio = np.concatenate(trozos)
    t = time.perf_counter() - t0

    destino = ruta("datos") / "prueba_mic.wav"
    sf.write(destino, audio, sr, subtype="PCM_16")
    pico = 20 * np.log10(np.abs(audio).max() / 32768 + 1e-9)
    saturadas = np.mean(np.abs(audio) >= 32767) * 100
    print(f"\nGuardado {destino.relative_to(RAIZ)}: {len(audio) / sr:.1f} s de audio recibidos en {t:.1f} s")
    print(f"Nivel medio {dbfs(audio):.1f} dBFS | pico {pico:.1f} dBFS | muestras saturadas {saturadas:.2f} %")
    if pico < -30:
        print("Muy bajo: revisa las conexiones o baja MIC_SHIFT en el config del firmware")
    elif saturadas > 0.1:
        print("Saturado: sube MIC_SHIFT en el config del firmware")


if __name__ == "__main__":
    main()
