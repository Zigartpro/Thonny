"""Fase 3.4: guarda la toma escogida del Audio 1 como referencia de F5-TTS.

Sin argumentos muestra las tomas de cada persona (duración y transcripción).
Con --hablante y --toma copia ese clip a datos/referencia_tts/<hablante>.wav y escribe su
transcripción exacta (lo que dijo, no el texto esperado) en <hablante>.txt.

Uso:  python pc/elegir_referencia_tts.py
      python pc/elegir_referencia_tts.py --hablante johan --toma 2
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, ruta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hablante")
    ap.add_argument("--toma", type=int)
    args = ap.parse_args()

    with open(ruta("clips_csv"), encoding="utf-8") as f:
        tomas = [r for r in csv.DictReader(f) if r["tipo"] == "miembro" and r["audio"] == "1"]

    if not args.hablante:
        for h in config()["hablantes"]:
            print(h)
            for r in (t for t in tomas if t["hablante"] == h):
                print(f"  toma {r['n']}: {r['duracion_s']} s  '{r['transcripcion_clip']}'")
        return

    elegida = next((r for r in tomas if r["hablante"] == args.hablante and int(r["n"]) == args.toma), None)
    if elegida is None:
        sys.exit(f"No hay toma {args.toma} del Audio 1 de {args.hablante} en clips.csv")
    destino = ruta("referencia_tts")
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RAIZ / elegida["ruta_celular"], destino / f"{args.hablante}.wav")
    texto = elegida["transcripcion_clip"] or elegida["transcripcion"]
    (destino / f"{args.hablante}.txt").write_text(texto + "\n", encoding="utf-8")
    print(f"{args.hablante}: toma {args.toma} ({elegida['duracion_s']} s) -> "
          f"{(destino / args.hablante).relative_to(RAIZ)}.wav/.txt\n  '{texto}'")


if __name__ == "__main__":
    main()
