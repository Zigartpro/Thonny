"""Evalúa las muestras de voz clonada (voces/muestras/) para ayudar a escoger la referencia.

Para cada muestra mide:
  - inteligibilidad: similitud (0-100) entre lo que transcribe Whisper y el texto pedido;
  - parecido: similitud coseno ECAPA entre la muestra y el promedio de la voz real de esa
    persona (sus clips de los audios 2-4, sin `test`), y contra las otras personas.
Resultado en resultados/evaluar_muestras_voz.json.

Uso:  python pc/evaluar_muestras_voz.py [--respuestas]
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_dlls  # noqa: E402,F401
from comun import RAIZ, config, ruta, similitud  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--respuestas", action="store_true",
                    help="evaluar las respuestas de voces/<hablante>/ en vez de las muestras")
    args = ap.parse_args()
    cfg = config()
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    H = cfg["hablante"]
    ecapa = EncoderClassifier.from_hparams(source=H["modelo"], savedir=str(RAIZ / H["carpeta"]),
                                           run_opts={"device": H["dispositivo"]},
                                           local_strategy=LocalStrategy.COPY)
    W = cfg["whisper"]
    whisper = WhisperModel(W["modelo"], device=W["dispositivo"], compute_type=W["compute_type"])

    def embedding(wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            import torchaudio.functional as AF
            wav = AF.resample(torch.from_numpy(wav), sr, 16000).numpy()
        e = ecapa.encode_batch(torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
        return e / np.linalg.norm(e)

    with open(ruta("clips_csv"), encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    centroides = {}
    for h in cfg["hablantes"]:
        embs = []
        for r in filas:
            if r["hablante"] == h and r["audio"] in ("2", "3", "4") and r["particion"] != "test":
                x, sr = sf.read(RAIZ / r["ruta_celular"], dtype="float32")
                embs.append(embedding(x, sr))
        c = np.mean(embs, axis=0)
        centroides[h] = c / np.linalg.norm(c)
        print(f"{h}: voz real promediada de {len(embs)} clips")

    with open(ruta("respuestas"), encoding="utf-8") as f:
        respuestas = yaml.safe_load(f)
    if args.respuestas:
        audios = [(ruta("voces") / h / f"{clave}.wav", h, texto)
                  for h in cfg["hablantes"] for clave, texto in respuestas["clonadas"].items()]
        audios = [a for a in audios if a[0].exists()]
    else:
        audios = [(p, p.stem.split("_toma")[0], respuestas["muestra"])
                  for p in sorted((ruta("voces") / "muestras").glob("*_toma*.wav"))]
    res = []
    print(f"\n{'audio':30s} {'inteligib.':>10s} {'parecido':>9s}  {'vs otros (máx)':>14s}")
    for p, h, texto in audios:
        x, sr = sf.read(p, dtype="float32")
        e = embedding(x, sr)
        segs, _ = whisper.transcribe(x if sr == 16000 else str(p), language=W["idioma"], beam_size=W["beam_size"])
        oido = " ".join(s.text.strip() for s in segs)
        intel = round(similitud(texto, oido))
        propio = float(e @ centroides[h])
        otros = {o: round(float(e @ c), 3) for o, c in centroides.items() if o != h}
        nombre = f"{h}/{p.name}" if args.respuestas else p.name
        print(f"{nombre:30s} {intel:10d} {propio:9.3f}  {max(otros.values()):14.3f}   '{oido}'")
        res.append({"audio": nombre, "hablante": h, "texto": texto, "inteligibilidad": intel,
                    "transcripcion": oido, "parecido_propio": round(propio, 3), "parecido_otros": otros})

    nombre = "evaluar_respuestas_voz.json" if args.respuestas else "evaluar_muestras_voz.json"
    salida = ruta("resultados") / nombre
    salida.write_text(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"),
                                  "audios": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
