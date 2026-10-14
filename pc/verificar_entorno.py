"""Fase 0: verifica el entorno de Thonny.

Muestra GPU, VRAM libre y versiones, y transcribe un WAV de prueba con faster-whisper
(modelo y compute_type de config.yaml), midiendo tiempo y VRAM usada.
El resultado queda en resultados/verificar_entorno.json.

Uso:  python pc/verificar_entorno.py [ruta.wav]
Si no se da un WAV y el de config.yaml no existe, se genera uno con gTTS
(necesita internet esa vez).
"""

import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cuda_dlls  # noqa: E402,F401  (antes de faster_whisper)
import torch  # noqa: E402

MB = 1024 * 1024
PAQUETES = ["torch", "torchaudio", "faster-whisper", "ctranslate2", "silero-vad",
            "speechbrain", "scikit-learn", "rapidfuzz", "pydub", "sounddevice",
            "soundfile", "pyyaml", "gTTS"]
TEXTO_PRUEBA = "Hola Thonny, prende la luz del cuarto y apaga el ventilador, por favor."


def cargar_config() -> dict:
    with open(RAIZ / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def versiones() -> dict:
    v = {"python": platform.python_version()}
    for p in PAQUETES:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    try:
        salida = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
        v["ffmpeg"] = salida.split("\n")[0]
    except FileNotFoundError:
        v["ffmpeg"] = None
    return v


def vram_libre_mb() -> float:
    libre, _ = torch.cuda.mem_get_info()
    return libre / MB


def generar_wav_gtts(destino: Path) -> None:
    """Genera el WAV de prueba con gTTS (necesita internet) y ffmpeg (16 kHz, mono)."""
    from gtts import gTTS

    destino.parent.mkdir(parents=True, exist_ok=True)
    mp3 = destino.with_suffix(".mp3")
    gTTS(TEXTO_PRUEBA, lang="es").save(str(mp3))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ar", "16000", "-ac", "1", str(destino)], check=True)
    mp3.unlink()


def main() -> None:
    cfg = cargar_config()
    cw = cfg["whisper"]
    res: dict = {"fecha": datetime.now().isoformat(timespec="seconds")}

    print("== Versiones ==")
    res["versiones"] = versiones()
    for k, val in res["versiones"].items():
        print(f"  {k:15s} {val}")
    faltan = [k for k, val in res["versiones"].items() if val is None]
    if faltan:
        print(f"  ¡Faltan: {', '.join(faltan)}!")

    print("\n== GPU ==")
    if not torch.cuda.is_available():
        sys.exit("CUDA no está disponible para PyTorch.")
    prop = torch.cuda.get_device_properties(0)
    torch.cuda.init()
    libre_base = vram_libre_mb()
    res["gpu"] = {
        "nombre": prop.name,
        "vram_total_mb": round(prop.total_memory / MB),
        "vram_libre_inicio_mb": round(libre_base),
        "cuda_torch": torch.version.cuda,
        "cudnn_torch": torch.backends.cudnn.version(),
    }
    for k, val in res["gpu"].items():
        print(f"  {k:22s} {val}")

    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / cfg["rutas"]["wav_prueba"]
    if not wav.exists():
        print(f"\nNo existe {wav}; generándolo con gTTS…")
        generar_wav_gtts(wav)

    import soundfile as sf
    from faster_whisper import WhisperModel

    duracion = sf.info(str(wav)).duration
    print(f"\n== faster-whisper: {cw['modelo']} / {cw['compute_type']} ==")
    print(f"  WAV: {wav} ({duracion:.1f} s)")

    t0 = time.perf_counter()
    modelo = WhisperModel(cw["modelo"], device=cw["dispositivo"], compute_type=cw["compute_type"])
    t_carga = time.perf_counter() - t0
    libre_cargado = vram_libre_mb()

    minimo = [libre_cargado]
    corriendo = True

    def muestrear():
        while corriendo:
            minimo[0] = min(minimo[0], vram_libre_mb())
            time.sleep(0.05)

    hilo = threading.Thread(target=muestrear, daemon=True)
    hilo.start()
    t0 = time.perf_counter()
    segmentos, info = modelo.transcribe(str(wav), language=cw["idioma"],
                                        initial_prompt=cw["initial_prompt"], beam_size=5)
    texto = " ".join(s.text.strip() for s in segmentos)  # la transcripción ocurre al iterar
    t_trans = time.perf_counter() - t0
    corriendo = False
    hilo.join()

    res["whisper"] = {
        "modelo": cw["modelo"],
        "compute_type": cw["compute_type"],
        "wav": str(wav),
        "duracion_audio_s": round(duracion, 2),
        "tiempo_carga_s": round(t_carga, 2),
        "tiempo_transcripcion_s": round(t_trans, 2),
        "factor_tiempo_real": round(t_trans / duracion, 3),
        "vram_modelo_mb": round(libre_base - libre_cargado),
        "vram_pico_mb": round(libre_base - minimo[0]),
        "idioma_detectado": info.language,
        "texto": texto,
    }
    for k, val in res["whisper"].items():
        print(f"  {k:24s} {val}")

    salida = RAIZ / cfg["rutas"]["resultados"] / "verificar_entorno.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(RAIZ)}")
    if shutil.which("ffmpeg") is None:
        print("Aviso: ffmpeg no está en el PATH de este proceso (reinicia la terminal).")


if __name__ == "__main__":
    main()
