"""Fase 3.3: grabación en vivo directo al INMP441 (por cable).

Muestra cada frase de `vivo` en guion_audios.yaml, avisa "HABLA AHORA", graba 4 s, la
transcribe con Whisper para comprobarla y deja repetirla. Guarda en
datos/crudos/vivo/<nombre>/<nombre>_v<NN>.wav y en datos/vivo.csv. Si se interrumpe, al
volver a correrlo sigue con la primera frase que falte.

Uso:  python pc/grabar_en_vivo.py --nombre cristian            (una de las 3 personas)
      python pc/grabar_en_vivo.py --nombre pedro --ajeno       (alguien que NO es de los 3)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_dlls  # noqa: E402,F401  (antes de faster_whisper)
from comun import RAIZ, config, normalizar, ruta, similitud  # noqa: E402
from esp32 import Cable  # noqa: E402

SEGUNDOS = 3
COLUMNAS = ["nombre", "tipo", "n", "texto", "indicacion", "ruta", "nivel_dbfs", "transcripcion", "similitud"]


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean((x / 32768.0) ** 2)) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True, help="nombre en minúscula, sin tildes ni espacios")
    ap.add_argument("--ajeno", action="store_true", help="la persona NO es una de las 3 autorizadas")
    args = ap.parse_args()

    cfg = config()
    nombre = normalizar(args.nombre).replace(" ", "_")
    tipo = "ajeno" if args.ajeno else "miembro"
    if tipo == "miembro" and nombre not in cfg["hablantes"]:
        sys.exit(f"'{nombre}' no está en config.yaml > hablantes {cfg['hablantes']}. Si no es de los 3, usa --ajeno.")
    with open(RAIZ / cfg["rutas"]["guion"], encoding="utf-8") as f:
        frases = yaml.safe_load(f)["vivo"][tipo]

    carpeta = ruta("crudos") / "vivo" / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    csv_path = ruta("datos") / "vivo.csv"
    filas = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            filas = [r for r in csv.DictReader(f)]
    hechas = {(r["nombre"], int(r["n"])) for r in filas}
    pendientes = [(k, fr) for k, fr in enumerate(frases, 1) if (nombre, k) not in hechas]
    if not pendientes:
        print(f"{nombre} ya grabó las {len(frases)} frases.")
        return

    print("Cargando Whisper…")
    from faster_whisper import WhisperModel
    W = cfg["whisper"]
    whisper = WhisperModel(W["modelo"], device=W["dispositivo"], compute_type=W["compute_type"])
    print("Conectando con la ESP32 por cable…")
    placa = Cable()
    sr = placa.sr

    print(f"\n=== {nombre} ({tipo}): faltan {len(pendientes)} de {len(frases)} frases ===")
    print("Habla a unos 20-40 cm del micrófono, salvo que diga 'lejos'.\n")
    try:
        for k, fr in pendientes:
            while True:
                print("-" * 60)
                print(f"[{k}/{len(frases)}]  Di:  «{fr['texto']}»   ({fr['indicacion']})")
                input("   Enter cuando estés listo… ")

                def avisar():
                    time.sleep(0.75)  # la placa descarta 0.6 s mientras el micrófono se asienta
                    print(f"   >>> HABLA AHORA ({SEGUNDOS} s) <<<", flush=True)

                audio = placa.grabar(SEGUNDOS, al_empezar=avisar)
                print("   Recibiendo… listo.")
                # sin initial_prompt: con audio corto Whisper a veces repite el prompt en vez de oír
                segs, _ = whisper.transcribe(audio.astype(np.float32) / 32768, language=W["idioma"],
                                             beam_size=W["beam_size"], condition_on_previous_text=False)
                oido = " ".join(s.text.strip() for s in segs)
                sim = round(similitud(fr["texto"], oido))
                nivel = dbfs(audio)
                pico = float(20 * np.log10(np.abs(audio).max() / 32768 + 1e-9))
                print(f"   Whisper oyó: «{oido}»  (parecido {sim}/100) | nivel {nivel:.1f} dBFS, pico {pico:.1f} dBFS")
                if pico > -0.5:
                    print("   Ojo: saturó (muy cerca o muy duro).")
                elif pico < -35:
                    print("   Ojo: se oye muy bajo.")
                r = input("   Enter = guardar y seguir | r = repetir | q = salir: ").strip().lower()
                if r == "q":
                    raise KeyboardInterrupt
                if r == "r":
                    continue
                destino = carpeta / f"{nombre}_v{k:02d}.wav"
                sf.write(destino, audio, sr, subtype="PCM_16")
                filas.append({"nombre": nombre, "tipo": tipo, "n": k, "texto": fr["texto"],
                              "indicacion": fr["indicacion"], "ruta": destino.relative_to(RAIZ).as_posix(),
                              "nivel_dbfs": f"{nivel:.1f}", "transcripcion": oido, "similitud": sim})
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=COLUMNAS)
                    w.writeheader()
                    w.writerows(filas)
                break
        print(f"\n¡Listo! {nombre} grabó todas sus frases.")
    except KeyboardInterrupt:
        print("\nPausado. Vuelve a correr el mismo comando para seguir donde quedó.")
    finally:
        placa.cerrar()


if __name__ == "__main__":
    main()
