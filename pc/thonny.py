"""Fase 6: Thonny completo.

Escucha -> corta frases (Silero VAD) -> texto (faster-whisper) -> ¿dice "Hola Thonny"? ->
¿quién habla? (ECAPA + modelos/hablante.joblib) -> intención (modelos/intenciones.joblib) ->
responde con la voz de esa persona (voces/<persona>/) y acciona el relé.

Entradas (--entrada):
  pc     micrófono del portátil, tiempo real (por defecto)
  wifi   INMP441 por WiFi (firmware main.py), tiempo real
  cable  INMP441 por el cable: Enter y hablar 3 s (no es tiempo real)
Salidas (--salida, por defecto igual a la entrada): pc (parlantes del portátil), wifi, cable (MAX98357)
Relés (--reles): simulados (por defecto, solo en pantalla), cable, wifi

Uso:  python pc/thonny.py
      python pc/thonny.py --entrada cable --reles cable
      python pc/thonny.py --archivo prueba.wav       (procesa un WAV como si lo hubieran dicho)
      python pc/thonny.py --sin-verificar             (modo depuración: no verifica quién habla)
"""

import argparse
import logging
import queue
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_dlls  # noqa: E402,F401  (antes de faster_whisper)
from comun import RAIZ, config, ruta, texto_intencion, tokens  # noqa: E402

SR = 16000
RELES = {"bombillo": 1, "ventilador": 2, "cautin": 3, "plancha": 4}


# ------------------------------------------------------------------ entradas

class EntradaPC:
    """Micrófono del portátil en tiempo real; entrega bloques de 512 muestras."""

    def __init__(self):
        import sounddevice as sd
        self.cola: queue.Queue = queue.Queue()
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=512,
                                     callback=lambda d, *_: self.cola.put(d[:, 0].copy()))
        self.stream.start()

    def bloque(self) -> np.ndarray:
        return self.cola.get()

    def vaciar(self):
        while not self.cola.empty():
            self.cola.get_nowait()


class EntradaWiFi:
    def __init__(self, host=None):
        from esp32 import Microfono
        self.mic = Microfono(host)

    def bloque(self) -> np.ndarray:
        return self.mic.leer(512)

    def vaciar(self):
        pass  # el audio que llega mientras Thonny habla se descarta al leer (ver Thonny.hablar)


class Segmentador:
    """Corta el audio continuo en frases con Silero VAD."""

    def __init__(self, entrada, max_s=8.0):
        from silero_vad import VADIterator, load_silero_vad
        import torch
        self.torch = torch
        self.vad = VADIterator(load_silero_vad(), threshold=0.5, sampling_rate=SR,
                               min_silence_duration_ms=500, speech_pad_ms=200)
        self.entrada = entrada
        self.max = int(max_s * SR)

    def frase(self) -> np.ndarray:
        previo, voz, hablando = [], [], False
        while True:
            b = self.entrada.bloque()
            ev = self.vad(self.torch.from_numpy(b.astype(np.float32) / 32768))
            if hablando:
                voz.append(b)
                if (ev and "end" in ev) or sum(map(len, voz)) >= self.max:
                    self.vad.reset_states()
                    return np.concatenate(voz)
            else:
                previo = (previo + [b])[-10:]  # ~0.3 s antes de que arranque la voz
                if ev and "start" in ev:
                    hablando, voz = True, list(previo)

    def reiniciar(self):
        self.vad.reset_states()
        self.entrada.vaciar()


# ------------------------------------------------------------------ Thonny

class Thonny:
    def __init__(self, args):
        cfg = config()
        self.cfg, self.args = cfg, args
        logs = RAIZ / "logs"
        logs.mkdir(exist_ok=True)
        logging.basicConfig(filename=logs / "thonny.log", level=logging.INFO, encoding="utf-8",
                            format="%(asctime)s %(message)s")
        self.log = logging.getLogger("thonny")

        print("Cargando modelos…")
        from faster_whisper import WhisperModel
        W = cfg["whisper"]
        self.W = W
        self.whisper = WhisperModel(W["modelo"], device=W["dispositivo"], compute_type=W["compute_type"])
        from hablante import Encoder
        self.enc = Encoder()
        V = cfg["verificacion"]
        h = joblib.load(RAIZ / V["modelo_final"]["archivo"])
        self.hablante, self.umbral = h["modelo"], h["umbral"]
        if V.get("umbral_thonny") is not None:
            self.umbral = float(V["umbral_thonny"])
        self.intenciones = joblib.load(RAIZ / cfg["intenciones"]["modelo"])
        self.umbral_intencion = cfg["intenciones"]["umbral_confianza"]
        with open(ruta("respuestas"), encoding="utf-8") as f:
            self.respuestas = yaml.safe_load(f)
        from preparar_audios import es_alucinacion
        self.es_alucinacion = es_alucinacion

        self.cable = None
        if "cable" in (args.entrada, args.salida, args.reles):
            from esp32 import Cable
            self.cable = Cable()
        self.control = None
        if "wifi" in (args.salida, args.reles):
            from esp32 import Control
            self.control = Control(args.host)
        self.voces_placa = self.cable.voces_en_placa() if args.salida == "cable" else set()
        if args.salida == "cable" and not self.voces_placa:
            print("Aviso: la placa no tiene las voces guardadas; cada respuesta tardará ~6 s. "
                  "Súbelas una vez con: python pc/subir_voces.py")
        if args.reles == "cable":
            self.cable.rele(1, False)  # deja todo en apagado al arrancar
            for n in range(2, 5):
                self.cable.rele(n, False)
        print(f"Umbral de hablante: {self.umbral:.3f} | entrada {args.entrada}, salida {args.salida}, "
              f"relés {args.reles}" + (" | SIN verificar hablante" if args.sin_verificar else ""))

    # -------------------------------------------------------------- piezas

    def transcribir(self, audio: np.ndarray) -> str:
        segs, _ = self.whisper.transcribe(audio.astype(np.float32) / 32768, language=self.W["idioma"],
                                          initial_prompt=self.W["initial_prompt"], beam_size=self.W["beam_size"],
                                          condition_on_previous_text=False)
        P = self.cfg["preparacion"]
        return " ".join(s.text.strip() for s in segs if not self.es_alucinacion(s, P))

    def quien(self, audio: np.ndarray) -> tuple[str, float]:
        e = self.enc.de_audio(audio.astype(np.float32) / 32768)
        personas, puntajes = self.hablante.puntuar(e[None, :])
        return str(personas[0]), float(puntajes[0])

    def intencion(self, texto: str) -> tuple[str | None, float]:
        t = texto_intencion(texto)
        if not t:
            return None, 0.0
        p = self.intenciones.predict_proba([t])[0]
        k = int(np.argmax(p))
        return str(self.intenciones.classes_[k]), float(p[k])

    def hablar(self, persona: str, clave: str):
        wav = ruta("voces") / persona / f"{clave}.wav"
        if not wav.exists():
            wav = ruta("voces") / "sistema" / f"{clave}.wav"
        if not wav.exists():
            print(f"   (falta el audio {persona}/{clave})")
            return
        x, sr = sf.read(wav, dtype="int16")
        print(f"   🔊 {persona}: «{self.respuestas['clonadas'].get(clave, self.respuestas.get('sistema', {}).get(clave, clave))}»")
        if self.args.salida == "pc":
            import sounddevice as sd
            sd.play(x, sr)
            sd.wait()
        elif self.args.salida == "cable":
            nombre = f"{wav.parent.name}/{clave}"
            if nombre in self.voces_placa:  # guardada en la flash (pc/subir_voces.py): inmediato
                self.cable.reproducir_voz(nombre)
            else:                           # si no, se transfiere (~6 s)
                self.cable.reproducir_cable(x)
        else:
            self.control.reproducir(x)

    def rele(self, aparato: str, encendido: bool):
        n = RELES[aparato]
        estado = "PRENDIDO" if encendido else "APAGADO"
        print(f"   💡 Relé {n} ({aparato}): {estado}")
        if self.args.reles == "cable":
            self.cable.rele(n, encendido)
        elif self.args.reles == "wifi":
            self.control.rele(n, encendido)

    def ejecutar(self, persona: str, intencion: str | None, confianza: float) -> bool:
        """Devuelve True si entendió un comando."""
        if intencion and "_" in intencion and confianza >= self.umbral_intencion:
            accion, aparato = intencion.split("_", 1)
            if aparato in RELES:
                self.rele(aparato, accion == "prender")
                self.hablar(persona, intencion)
                return True
        return False

    # -------------------------------------------------------------- una frase

    def procesar(self, audio: np.ndarray, esperando: str | None = None) -> str | None:
        """Procesa una frase. `esperando` = persona a la que Thonny le preguntó "¿Qué necesitas?".
        Devuelve la persona si Thonny queda esperando su comando."""
        t0 = time.perf_counter()
        texto = self.transcribir(audio)
        if not texto:
            return esperando
        toks = tokens(texto)
        activado = "thonny" in toks
        print(f"\n🗣  «{texto}»  ({len(audio) / SR:.1f} s)")
        if not activado and esperando is None:
            print("   (sin 'Hola Thonny': ignorado)")
            return None

        if self.args.sin_verificar:
            persona, puntaje = (esperando or self.cfg["hablantes"][0]), 1.0
        elif esperando is not None and not activado and not self.cfg["verificacion"].get("verificar_comando", True):
            # el "Hola Thonny" ya verificó a esta persona; un comando corto da puntajes poco fiables
            persona, puntaje = esperando, 1.0
        else:
            persona, puntaje = self.quien(audio)
        aceptado = puntaje >= self.umbral
        print(f"   👤 parece {persona} (puntaje {puntaje:.3f}, umbral {self.umbral:.3f}) -> "
              f"{'ACEPTADO' if aceptado else 'RECHAZADO'}")
        self.log.info(f"texto={texto!r} persona={persona} puntaje={puntaje:.3f} aceptado={aceptado}")
        if not aceptado:
            print("   🚫 Voz no autorizada: Thonny no se activa.")
            return None
        if esperando is not None and persona != esperando and not activado:
            print(f"   🚫 Thonny esperaba a {esperando}, no a {persona}.")
            return None

        # comando: lo que viene después del último "thonny" (o todo, si ya estaba esperando)
        resto = toks[len(toks) - 1 - toks[::-1].index("thonny") + 1:] if activado else toks
        intencion, conf = self.intencion(" ".join(resto))
        if intencion:
            print(f"   🎯 intención: {intencion} (confianza {conf:.2f})")
        self.log.info(f"persona={persona} intencion={intencion} confianza={conf:.2f}")
        if self.ejecutar(persona, intencion, conf):
            print(f"   ({time.perf_counter() - t0:.1f} s)")
            return None
        if not resto or intencion == "saludo":
            self.hablar(persona, "que_necesitas")
            return persona  # queda escuchando el comando de esta persona
        self.hablar(persona, "no_entendi")
        return None

    # -------------------------------------------------------------- bucles

    def correr(self):
        if self.args.archivo:
            x, sr = sf.read(self.args.archivo, dtype="int16")
            assert sr == SR, "el WAV debe estar a 16 kHz"
            self.procesar(x if x.ndim == 1 else x[:, 0])
            return
        if self.args.entrada == "cable":
            return self.correr_cable()
        entrada = EntradaPC() if self.args.entrada == "pc" else EntradaWiFi(self.args.host)
        seg = Segmentador(entrada)
        self.hablar("sistema", "listo")
        print("\n=== Thonny escuchando. Di «Hola Thonny, prende la luz». Ctrl+C para salir ===")
        esperando, hasta = None, 0.0
        while True:
            frase = seg.frase()
            if esperando and time.monotonic() > hasta:
                esperando = None
            nuevo = self.procesar(frase, esperando)
            seg.reiniciar()  # descarta lo que se oyó mientras Thonny hablaba
            esperando, hasta = nuevo, time.monotonic() + 6

    def correr_cable(self):
        print("\n=== Thonny por cable: Enter y di la frase (3 s). Ctrl+C para salir ===")
        esperando = None
        while True:
            input("\nEnter y habla…")
            audio = self.cable.grabar(3, al_empezar=lambda: (time.sleep(0.75), print(">>> HABLA AHORA <<<")))
            esperando = self.procesar(audio, esperando)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", choices=["pc", "wifi", "cable"], default="pc")
    ap.add_argument("--salida", choices=["pc", "wifi", "cable"])
    ap.add_argument("--reles", choices=["simulados", "cable", "wifi"], default="simulados")
    ap.add_argument("--host", help="IP de la ESP32 (modo wifi)")
    ap.add_argument("--archivo", help="procesar un WAV de 16 kHz en vez de escuchar")
    ap.add_argument("--sin-verificar", action="store_true", help="depuración: no verifica quién habla")
    args = ap.parse_args()
    args.salida = args.salida or (args.entrada if args.entrada != "cable" else "pc")
    thonny = Thonny(args)
    try:
        thonny.correr()
    except KeyboardInterrupt:
        print("\nThonny apagado.")
    finally:
        if thonny.cable:
            if args.reles == "cable":
                for n in range(1, 5):
                    thonny.cable.rele(n, False)
            thonny.cable.cerrar()


if __name__ == "__main__":
    main()
