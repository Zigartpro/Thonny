"""Fase 5: genera las respuestas de Thonny con la voz clonada de cada persona (F5-TTS).

Corre en el venv de TTS y con Thonny cerrado (VRAM):
    C:\\Users\\duvan\\venvs\\thonny-tts\\Scripts\\python.exe pc\\generar_voces.py --muestra

  --muestra   Para cada persona y cada toma de su Audio 1 (clips a1 de clips.csv) genera la frase
              de muestra de respuestas.yaml en voces/muestras/<hablante>_toma<N>.wav, para
              escoger de oído la mejor referencia.
  (sin bandera) Genera todas las respuestas de respuestas.yaml para cada persona, usando su
              referencia escogida en datos/referencia_tts/<hablante>.wav + .txt, en
              voces/<hablante>/<clave>.wav (16 kHz mono, lo que reproduce la ESP32). Cada frase se
              verifica con Whisper y se reintenta con otra semilla si no se entiende bien.

Mide tiempo y VRAM y guarda el resumen en resultados/generar_voces[_muestra].json.
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import yaml
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, canonica, config, ruta  # noqa: E402

MB = 1024 * 1024


def texto_pronunciable(texto: str) -> str:
    """Cambia cada variante de "Thonny" (Tommy, Tony…) por el nombre pronunciable."""
    nombre = config()["tts"]["nombre_pronunciable"]
    return re.sub(r"[\wáéíóúñü]+", lambda m: nombre if canonica(m.group(0)) == "thonny" else m.group(0),
                  texto, flags=re.IGNORECASE)


def cargar_modelo():
    from f5_tts.api import F5TTS

    T = config()["tts"]
    ckpt = hf_hub_download(T["repo"], T["checkpoint"])
    vocab = hf_hub_download(T["repo"], T["vocab"])
    return F5TTS(model=T["arquitectura"], ckpt_file=ckpt, vocab_file=vocab, device="cuda")


def sintetizar(modelo, ref_wav: Path, ref_texto: str, texto: str, semilla: int | None = None) -> tuple[np.ndarray, int, float]:
    T = config()["tts"]
    t0 = time.perf_counter()
    wav, sr, _ = modelo.infer(ref_file=str(ref_wav), ref_text=ref_texto, gen_text=texto,
                              nfe_step=T["nfe_step"], speed=T["velocidad"],
                              seed=T["semilla"] if semilla is None else semilla,
                              show_info=lambda *a, **k: None)
    return np.asarray(wav, dtype=np.float32), sr, time.perf_counter() - t0


def a_salida(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    sr_out = config()["tts"]["sample_rate_salida"]
    if sr != sr_out:
        wav = AF.resample(torch.from_numpy(wav), sr, sr_out).numpy()
    return wav, sr_out


def tomas_audio1(hablante: str) -> list[dict]:
    with open(ruta("clips_csv"), encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r["hablante"] == hablante and r["tipo"] == "miembro" and r["audio"] == "1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--muestra", action="store_true", help="una muestra por cada toma del Audio 1")
    ap.add_argument("--hablante", action="append", help="solo este hablante (se puede repetir)")
    ap.add_argument("--clave", action="append", help="solo esta respuesta de respuestas.yaml (se puede repetir)")
    ap.add_argument("--semilla", type=int, help="otra semilla, para regenerar una frase que salió mal")
    args = ap.parse_args()

    cfg = config()
    with open(ruta("respuestas"), encoding="utf-8") as f:
        respuestas = yaml.safe_load(f)
    hablantes = [h for h in cfg["hablantes"] if not args.hablante or h in args.hablante]

    torch.cuda.init()
    libre_inicio = torch.cuda.mem_get_info()[0] / MB
    t0 = time.perf_counter()
    modelo = cargar_modelo()
    t_carga = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    print(f"Modelo cargado en {t_carga:.1f} s | VRAM libre al empezar {libre_inicio:.0f} MB")

    generados = []
    if args.muestra:
        destino = ruta("voces") / "muestras"
        destino.mkdir(parents=True, exist_ok=True)
        texto = respuestas["muestra"]
        for h in hablantes:
            tomas = tomas_audio1(h)
            if not tomas:
                print(f"{h}: no hay clips del Audio 1 en clips.csv")
            for r in tomas:
                ref_wav = RAIZ / r["ruta_celular"]
                ref_texto = texto_pronunciable(r["transcripcion_clip"] or r["frase_esperada"])
                wav, sr, t = sintetizar(modelo, ref_wav, ref_texto, texto)
                salida = destino / f"{h}_toma{int(r['n'])}.wav"
                sf.write(salida, wav, sr, subtype="PCM_16")
                dur = len(wav) / sr
                print(f"  {salida.relative_to(RAIZ)}  {dur:.1f} s de audio en {t:.1f} s")
                generados.append({"hablante": h, "toma": int(r["n"]), "referencia": r["ruta_celular"],
                                  "texto_referencia": ref_texto, "texto": texto, "salida": str(salida.relative_to(RAIZ)),
                                  "duracion_s": round(dur, 2), "tiempo_s": round(t, 2)})
    else:
        T = cfg["tts"]
        import cuda_dlls  # noqa: F401  (antes de faster_whisper)
        from faster_whisper import WhisperModel
        from comun import similitud
        W = cfg["whisper"]
        whisper = WhisperModel(W["modelo"], device=W["dispositivo"], compute_type=W["compute_type"])

        def verificar(wav: np.ndarray, texto: str) -> tuple[str, int]:
            segs, _ = whisper.transcribe(wav, language=W["idioma"], beam_size=W["beam_size"])
            oido = " ".join(s.text.strip() for s in segs)
            return oido, round(similitud(texto, oido))

        ref_dir = ruta("referencia_tts")
        for h in hablantes:
            ref_wav, ref_txt = ref_dir / f"{h}.wav", ref_dir / f"{h}.txt"
            if not (ref_wav.exists() and ref_txt.exists()):
                print(f"{h}: falta {ref_wav.relative_to(RAIZ)} o su .txt (escoge la referencia primero)")
                continue
            ref_texto = texto_pronunciable(ref_txt.read_text(encoding="utf-8").strip())
            carpeta = ruta("voces") / h
            carpeta.mkdir(parents=True, exist_ok=True)
            for clave, texto in respuestas["clonadas"].items():
                if args.clave and clave not in args.clave:
                    continue
                # Genera, verifica con Whisper y reintenta con otra semilla si no se entiende bien
                semilla0 = cfg["tts"]["semilla"] if args.semilla is None else args.semilla
                mejor, intentos = None, []
                for k in range(T["intentos"]):
                    semilla = semilla0 + k
                    wav, sr, t = sintetizar(modelo, ref_wav, ref_texto, texto, semilla)
                    wav, sr = a_salida(wav, sr)
                    oido, intel = verificar(wav, texto)
                    intentos.append({"semilla": semilla, "inteligibilidad": intel, "transcripcion": oido})
                    if mejor is None or intel > mejor[0]:
                        mejor = (intel, semilla, wav, sr, oido)
                    if intel >= T["inteligibilidad_min"]:
                        break
                intel, semilla, wav, sr, oido = mejor
                salida = carpeta / f"{clave}.wav"
                sf.write(salida, wav, sr, subtype="PCM_16")
                aviso = "" if intel >= T["inteligibilidad_min"] else "  <- ESCUCHAR, no llegó al mínimo"
                print(f"  {salida.relative_to(RAIZ)}  intel {intel} (semilla {semilla}, "
                      f"{len(intentos)} intento/s) '{oido}'{aviso}")
                generados.append({"hablante": h, "clave": clave, "texto": texto, "semilla": semilla,
                                  "inteligibilidad": intel, "transcripcion": oido, "intentos": intentos,
                                  "salida": str(salida.relative_to(RAIZ))})

    pico = torch.cuda.max_memory_allocated() / MB
    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modo": "muestra" if args.muestra else "completo",
        "tts": cfg["tts"],
        "tiempo_carga_s": round(t_carga, 1),
        "vram_libre_inicio_mb": round(libre_inicio),
        "vram_pico_torch_mb": round(pico),
        "generados": generados,
    }
    nombre = "generar_voces_muestra.json" if args.muestra else "generar_voces.json"
    salida = ruta("resultados") / nombre
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVRAM pico (PyTorch): {pico:.0f} MB | resumen en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
