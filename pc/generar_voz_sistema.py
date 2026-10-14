"""Fase 5.4: genera la voz por defecto (gTTS, necesita internet una vez) para los mensajes del
sistema de respuestas.yaml > sistema, en voces/sistema/<clave>.wav (16 kHz mono).

Uso:  python pc/generar_voz_sistema.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, ruta  # noqa: E402


def main() -> None:
    from gtts import gTTS

    with open(ruta("respuestas"), encoding="utf-8") as f:
        sistema = yaml.safe_load(f)["sistema"]
    destino = ruta("voces") / "sistema"
    destino.mkdir(parents=True, exist_ok=True)
    sr = config()["tts"]["sample_rate_salida"]
    for clave, texto in sistema.items():
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / "x.mp3"
            gTTS(texto, lang="es").save(str(mp3))
            salida = destino / f"{clave}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", str(sr), str(salida)],
                           check=True)
        print(f"{salida.relative_to(RAIZ)}  «{texto}»")


if __name__ == "__main__":
    main()
