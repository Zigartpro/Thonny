"""Guarda las respuestas (voces/<persona>/*.wav y voces/sistema/) en la flash de la ESP32, por el
cable, para que Thonny las reproduzca al instante en vez de transferir cada una (~6 s).
Se corre una sola vez (o cuando cambien las voces); las que ya están en la placa se saltan.

Uso:  python pc/subir_voces.py            (solo las que faltan)
      python pc/subir_voces.py --todas    (vuelve a subir todas)
"""
import argparse
import time

import soundfile as sf

from comun import config, ruta
from esp32 import Cable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--todas", action="store_true", help="sube también las que ya están en la placa")
    args = ap.parse_args()

    cfg = config()
    wavs = []
    for persona in [*cfg["hablantes"], "sistema"]:
        wavs += sorted((ruta("voces") / persona).glob("*.wav"))

    placa = Cable()
    try:
        ya = set() if args.todas else placa.voces_en_placa()
        pendientes = [w for w in wavs if f"{w.parent.name}/{w.stem}" not in ya]
        print(f"{len(wavs)} respuestas, {len(wavs) - len(pendientes)} ya en la placa, subiendo {len(pendientes)}…")
        t0 = time.time()
        for i, w in enumerate(pendientes, 1):
            x, sr = sf.read(w, dtype="int16")
            assert sr == 16000 and x.ndim == 1, f"{w} debe ser 16 kHz mono"
            nombre = f"{w.parent.name}/{w.stem}"
            placa.guardar_voz(nombre, x)
            print(f"  [{i}/{len(pendientes)}] {nombre} ({len(x) / sr:.1f} s)")
        print(f"Listo en {time.time() - t0:.0f} s. En la placa: {len(placa.voces_en_placa())} respuestas.")
    finally:
        placa.cerrar()


if __name__ == "__main__":
    main()
