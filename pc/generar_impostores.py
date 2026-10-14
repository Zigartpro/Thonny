"""Impostores sintéticos: voces ajenas (OpenSLR SLR72) clonadas con F5-TTS diciendo
"Hola Thonny" y comandos, para probar el rechazo con la frase real de activación.

Para cada hablante público toma como referencia una de sus frases originales (6-11 s, audio
de estudio), genera las frases de `ajenos.frases_impostor` y las pasa por el mismo canal tipo
WhatsApp que el resto (pc/preparar_ajenos.py). Llena datos/ajenos_sinteticos.csv; cada clip
hereda la partición de su hablante (pc/particion.py).

Corre en el venv de TTS:
    C:\\Users\\duvan\\venvs\\thonny-tts\\Scripts\\python.exe pc\\generar_impostores.py
"""

import csv
import sys
import tempfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, ruta  # noqa: E402
from generar_voces import cargar_modelo, sintetizar, texto_pronunciable  # noqa: E402
from preparar_ajenos import pasar_por_canal  # noqa: E402

COLUMNAS = ["hablante", "tipo", "sexo", "ruta", "texto", "canal", "duracion_s", "particion"]


def main() -> None:
    cfg = config()
    base = Path(cfg["rutas"]["dataset_ajenos"])
    with open(ruta("datos") / "ajenos.csv", encoding="utf-8") as f:
        publicos = list(csv.DictReader(f))
    textos = {}
    for tsv in base.glob("line_index_*.tsv"):
        for linea in tsv.read_text(encoding="utf-8").splitlines():
            if "\t" in linea:
                k, t = linea.split("\t", 1)
                textos[k.strip()] = t.strip()

    modelo = cargar_modelo()
    destino = base / "sinteticos"
    destino.mkdir(exist_ok=True)
    frases = cfg["ajenos"]["frases_impostor"]
    canales = {c["nombre"]: c for c in cfg["ajenos"]["canales"]}
    filas = []
    hablantes = sorted({r["hablante"] for r in publicos})
    for h in hablantes:
        propias = [r for r in publicos if r["hablante"] == h]
        # referencia: una frase original de 6-11 s (la más larga dentro del rango)
        candidatas = []
        for r in propias:
            original = base / "wav" / Path(r["ruta"]).name
            d = sf.info(str(original)).duration
            if 6 <= d <= 11 and Path(r["ruta"]).stem in textos:
                candidatas.append((d, original, textos[Path(r["ruta"]).stem]))
        if not candidatas:
            print(f"{h}: sin referencia de 6-11 s, se omite")
            continue
        _, ref_wav, ref_texto = max(candidatas)
        for k, frase in enumerate(frases, 1):
            salida = destino / f"{h}_imp{k:02d}.wav"
            if not salida.exists():
                wav, sr, _ = sintetizar(modelo, ref_wav, ref_texto, texto_pronunciable(frase), semilla=k)
                with tempfile.TemporaryDirectory() as tmp:
                    crudo = Path(tmp) / "x.wav"
                    sf.write(crudo, wav, sr)
                    pasar_por_canal(crudo, salida, canales[propias[0]["canal"]],
                                    cfg["preparacion"]["sample_rate"], cfg["preparacion"])
            filas.append({"hablante": h, "tipo": "ajeno_sintetico", "sexo": propias[0]["sexo"],
                          "ruta": salida.as_posix(), "texto": frase, "canal": propias[0]["canal"],
                          "duracion_s": f"{sf.info(str(salida)).duration:.2f}", "particion": ""})
        print(f"  {h}: {len(frases)} frases")

    csv_path = ruta("datos") / "ajenos_sinteticos.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas)} impostores -> {csv_path.relative_to(RAIZ)} (corre pc/particion.py)")


if __name__ == "__main__":
    main()
