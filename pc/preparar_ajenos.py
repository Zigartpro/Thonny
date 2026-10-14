"""Prepara las voces ajenas públicas (OpenSLR SLR72, español colombiano, CC BY-SA 4.0).

Descomprime los zip del dataset (si hace falta), toma hasta `ajenos_max_clips` frases por
hablante (semilla fija) y las pasa por el mismo camino que los audios de los miembros
(compresión tipo WhatsApp, 16 kHz mono, normalización de volumen). Así el modelo no puede
distinguir "estudio" de "celular" en vez de distinguir voces. Llena datos/ajenos.csv.
Las particiones las asigna después pc/particion.py.

Uso:  python pc/preparar_ajenos.py
"""

import csv
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, ruta  # noqa: E402

COLUMNAS = ["hablante", "tipo", "sexo", "ruta", "texto", "canal", "duracion_s", "particion"]


def pasar_por_canal(origen: Path, salida: Path, canal: dict, sr: int, P: dict) -> None:
    """Mismo camino que los audios de los miembros: compresión tipo WhatsApp (Opus/AAC),
    luego 16 kHz mono y la misma normalización de volumen que pc/preparar_audios.py."""
    import tempfile

    import numpy as np

    from preparar_audios import normalizar_volumen

    with tempfile.TemporaryDirectory() as tmp:
        comprimido = Path(tmp) / f"x.{canal['ext']}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(origen), "-ac", "1", "-ar", str(canal["sr"]),
                        "-c:a", canal["codec"], "-b:a", canal["bitrate"], str(comprimido)], check=True)
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(comprimido), "-ac", "1", "-ar", str(sr),
                              "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, dtype=np.float32)
    sf.write(salida, normalizar_volumen(x, P), sr, subtype="PCM_16")


def main() -> None:
    cfg = config()
    base = Path(cfg["rutas"]["dataset_ajenos"])
    P = cfg["particion"]
    sr = cfg["preparacion"]["sample_rate"]

    crudo = base / "wav"
    for z in sorted(base.glob("es_co_*.zip")):
        marca = crudo / f".{z.stem}.ok"
        if not marca.exists():
            print(f"Descomprimiendo {z.name}…")
            crudo.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                for m in zf.infolist():  # solo .wav, sin rutas raras
                    if m.filename.endswith(".wav") and ".." not in m.filename:
                        (crudo / Path(m.filename).name).write_bytes(zf.read(m))
            marca.touch()

    textos = {}
    for tsv in base.glob("line_index_*.tsv"):
        for linea in tsv.read_text(encoding="utf-8").splitlines():
            if "\t" in linea:
                k, t = linea.split("\t", 1)
                textos[k.strip()] = t.strip()

    por_hablante: dict[str, list[Path]] = {}
    for w in sorted(crudo.glob("*.wav")):
        por_hablante.setdefault(w.stem.rsplit("_", 1)[0], []).append(w)

    rng = random.Random(P["semilla"])
    destino = base / "16k_canal"
    destino.mkdir(exist_ok=True)
    canales = cfg["ajenos"]["canales"]
    filas = []
    for k, (h, wavs) in enumerate(sorted(por_hablante.items())):
        canal = canales[k % len(canales)]  # cada hablante con un códec, alternando
        for w in sorted(rng.sample(wavs, min(P["ajenos_max_clips"], len(wavs)))):
            salida = destino / w.name
            if not salida.exists():
                pasar_por_canal(w, salida, canal, sr, cfg["preparacion"])
            filas.append({"hablante": h, "tipo": "ajeno_publico", "sexo": "f" if h.startswith("cof") else "m",
                          "ruta": salida.as_posix(), "texto": textos.get(w.stem, ""), "canal": canal["nombre"],
                          "duracion_s": f"{sf.info(str(salida)).duration:.2f}", "particion": ""})

    csv_path = ruta("datos") / "ajenos.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNAS)
        wr.writeheader()
        wr.writerows(filas)
    print(f"{len(por_hablante)} hablantes, {len(filas)} clips -> {csv_path.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
