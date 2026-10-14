"""Conexión con la ESP32 (protocolo en PROTOCOLO.md): audio del micrófono y control."""

import json
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import config  # noqa: E402


def _host(host: str | None) -> str:
    return host or config()["esp32"]["host"]


def quitar_continua(pcm16: np.ndarray, sr: int = 16000, corte_hz: float = 50.0) -> np.ndarray:
    """Pasa-altos (Butterworth de 2.º orden) que quita el corrimiento de continua del INMP441."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, corte_hz, "highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, pcm16.astype(np.float64))
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


class Microfono:
    """Recibe el audio continuo del INMP441 (PCM 16 bits LE, 16 kHz, mono)."""

    def __init__(self, host: str | None = None, timeout: float = 5.0):
        E = config()["esp32"]
        self.sr = E["sample_rate"]
        self.sock = socket.create_connection((_host(host), E["puerto_audio"]), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._resto = b""

    def leer(self, n_muestras: int) -> np.ndarray:
        """Devuelve exactamente n_muestras como int16."""
        necesito = n_muestras * 2
        datos = self._resto
        while len(datos) < necesito:
            trozo = self.sock.recv(max(4096, necesito - len(datos)))
            if not trozo:
                raise ConnectionError("la ESP32 cerró la conexión de audio")
            datos += trozo
        self._resto = datos[necesito:]
        return np.frombuffer(datos[:necesito], dtype="<i2").copy()

    def cerrar(self):
        self.sock.close()


class Cable:
    """La ESP32 por cable USB (Serial). Atiende las mismas órdenes JSON que el puerto de
    control, y con `modo_audio()` el cable pasa a llevar solo el audio del micrófono."""

    def __init__(self, puerto: str | None = None):
        import serial

        E = config()["esp32"]
        self.sr = E["sample_rate"]
        self.ser = serial.Serial()
        self.ser.port = puerto or E["puerto_serie"]
        self.ser.baudrate = E["baudios"]
        self.ser.timeout = 1
        self.ser.dtr = False  # abrir el puerto sin reiniciar la placa
        self.ser.rts = False
        self.ser.open()
        self._resto = b""

    def reiniciar(self):
        """Pulso en EN (RTS) para reiniciar la placa en modo normal."""
        import time
        self.ser.rts = True
        time.sleep(0.1)
        self.ser.rts = False
        time.sleep(2.5)
        self.ser.reset_input_buffer()

    def _esperar_respuesta(self, segundos: float) -> dict | None:
        import time
        fin = time.monotonic() + segundos
        while time.monotonic() < fin:
            linea = self.ser.readline().strip()
            if linea.startswith(b"{"):
                try:
                    return json.loads(linea)
                except json.JSONDecodeError:
                    pass
        return None

    def orden(self, d: dict, extra: bytes = b"", espera: float = 3.0) -> dict:
        self.ser.write(json.dumps(d).encode() + b"\n" + extra)
        r = self._esperar_respuesta(espera)
        if r is None:
            raise TimeoutError(f"la ESP32 no respondió a {d['cmd']} por el cable")
        return r

    def modo_audio(self):
        """Pide el audio por el cable. Si la placa ya estaba mandando audio (no responde),
        la reinicia y vuelve a pedirlo."""
        if config()["esp32"]["firmware"] == "micropython":
            return self._modo_audio_micropython()
        for intento in range(2):
            self.ser.reset_input_buffer()
            self.ser.write(b'{"cmd":"audio_serial"}\n')
            r = self._esperar_respuesta(3)
            if r and r.get("modo") == "audio_serial":
                return
            self.reiniciar()
        raise TimeoutError("la ESP32 no entró en modo audio por el cable")

    MARCA_AUDIO = b"THONNY_AUDIO\n"

    def _modo_audio_micropython(self):
        """Firmware MicroPython: por el REPL (115200) se corre `import cable`; la placa sube el
        cable a `baudios`, manda MARCA_AUDIO y desde ahí solo audio (firmware/micropython/cable.py)."""
        import time
        for intento in range(2):
            self.ser.baudrate = 115200
            self.ser.write(b"\x03\x03")  # detiene main.py
            time.sleep(0.3)
            self.ser.write(b"\x01")      # REPL crudo
            time.sleep(0.3)
            self.ser.reset_input_buffer()
            self.ser.write(b"import cable\x04")
            self.ser.flush()
            self.ser.baudrate = config()["esp32"]["baudios"]
            datos, fin = b"", time.monotonic() + 6
            while self.MARCA_AUDIO not in datos and time.monotonic() < fin:
                datos += self.ser.read(512)
            if self.MARCA_AUDIO in datos:
                self._resto = datos.split(self.MARCA_AUDIO, 1)[1]
                self._en_modo_cable = True
                return
            self.reiniciar()  # quizá ya estaba en modo audio: reiniciar y reintentar
        raise TimeoutError("la ESP32 no entró en modo audio por el cable")

    def _repl_listo(self):
        """Deja la placa (MicroPython) en el REPL crudo con firmware/micropython/grabar.py importado."""
        import time
        if getattr(self, "_repl", False):
            return
        self.ser.baudrate = 115200
        # Insiste con Ctrl-C (detiene main.py) y Ctrl-B (sale del REPL crudo si quedó en él) hasta
        # ver el prompt ">>> ". Así funciona aunque la placa esté arrancando en ese momento.
        datos, fin = b"", time.monotonic() + 15
        while b">>> " not in datos:
            if time.monotonic() > fin:
                raise TimeoutError(f"la placa no responde por el cable: {datos[-200:]!r}")
            self.ser.write(b"\x03\x02")
            time.sleep(0.25)
            datos += self.ser.read(self.ser.in_waiting or 1)
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.write(b"\x01")      # REPL crudo
        self._hasta(b"raw REPL; CTRL-B to exit\r\n>", 5)
        # reinicio suave desde el REPL crudo: libera la RAM que dejó main.py y no lo vuelve a correr
        self.ser.write(b"\x04")
        self._hasta(b"raw REPL; CTRL-B to exit\r\n>", 10)
        # el WiFi (lo prende main.py al arrancar) no se usa por cable y su consumo, sumado al del
        # amplificador, hacía reiniciar la placa por caída de voltaje (brownout)
        self.ser.write(b"import gc, network, grabar, parlante, reles, ubinascii; "
                       b"network.WLAN(network.STA_IF).active(False); gc.collect()\x04")
        self._hasta(b"\x04>", 10)
        self._repl = self._en_modo_cable = True

    def _hasta(self, marca: bytes, segundos: float) -> bytes:
        import time
        datos, fin = b"", time.monotonic() + segundos
        while marca not in datos:
            if time.monotonic() > fin:
                raise TimeoutError(f"la ESP32 no respondió por el cable (esperaba {marca!r}): {datos[-200:]!r}")
            datos += self.ser.read(1)
        return datos

    def grabar(self, segundos: float, al_empezar=None) -> np.ndarray:
        """Firmware MicroPython: la placa graba `segundos` en su RAM y después los manda por el
        cable (no es tiempo real; transferir tarda ~3 veces lo grabado). Máximo 3 s."""
        ms = int(segundos * 1000)
        assert 0 < ms <= 3000, "máximo 3 s por grabación (RAM de la placa)"
        self._repl_listo()
        self.ser.write(f"grabar.grabar({ms})".encode() + b"\x04")
        if al_empezar:
            al_empezar()  # p. ej. avisar "HABLA AHORA" cuando la placa empieza a grabar
        self._hasta(b"THONNY_PCM ", 10)
        n = int(self._hasta(b"\n", 2).strip())
        datos = b""
        while len(datos) < n:
            trozo = self.ser.read(n - len(datos))
            if not trozo:
                raise TimeoutError("se cortó la transferencia del audio")
            datos += trozo
        self._hasta(b"\x04>", 5)
        return quitar_continua(np.frombuffer(datos, dtype="<i2"), self.sr)

    def _ejecutar(self, codigo: str, espera: float = 10) -> str:
        """Ejecuta una línea en el REPL crudo de la placa y devuelve lo que imprimió."""
        self._repl_listo()
        self.ser.write(codigo.encode() + b"\x04")
        salida = self._hasta(b"\x04>", espera).decode("utf-8", "replace")
        if "Traceback" in salida:
            raise RuntimeError(f"error en la placa: {salida}")
        return salida

    def rele(self, n: int, encendido: bool) -> None:
        """Firmware MicroPython: prende/apaga el relé n (1-4) por el cable."""
        self._ejecutar(f"reles.poner({int(n)}, {bool(encendido)})")

    def tono(self, hz: int = 440, ms: int = 1000, volumen: float = 0.3) -> None:
        """Firmware MicroPython: tono de prueba generado en la placa (no transfiere audio)."""
        self._ejecutar(f"parlante.tono({hz}, {ms}, {volumen})", espera=ms / 1000 + 5)

    def reproducir_cable(self, pcm16: np.ndarray, al_terminar_envio=None) -> None:
        """Firmware MicroPython: manda el audio crudo a la RAM de la placa (~10 KB/s) y lo
        reproduce. Usa los bloques ya reservados de grabar.py (0.5 s cada uno), así que el audio
        se recorta a 3 s: pedir un bytearray nuevo fallaba por memoria (MemoryError)."""
        bloque, trozo, max_s = 8000 * 2, 1024, 3
        datos = np.asarray(pcm16, dtype="<i2").tobytes()[:self.sr * 2 * max_s]
        self._repl_listo()
        self.ser.write(f"parlante.recibir({len(datos)}, {trozo})".encode() + b"\x04")
        self._hasta(b"LISTO\r\n", 5)
        # mismos trozos que parlante.recibir: de a 1024 bytes sin cruzar de un bloque a otro;
        # la placa contesta "K" por cada uno, así su búfer de entrada nunca se desborda
        for inicio in range(0, len(datos), bloque):
            fin_bloque = min(inicio + bloque, len(datos))
            for i in range(inicio, fin_bloque, trozo):
                self.ser.write(datos[i:min(i + trozo, fin_bloque)])
                self._hasta(b"K", 5)
        self._hasta(b"\x04>", 5)
        if al_terminar_envio:
            al_terminar_envio()
        self._ejecutar(f"parlante.reproducir_bloques({len(datos)}); gc.collect()",
                       espera=len(datos) / (self.sr * 2) + 5)

    def guardar_voz(self, nombre: str, pcm16: np.ndarray) -> None:
        """Firmware MicroPython: guarda una respuesta en la flash de la placa (/voces/<nombre>.pcm,
        p. ej. nombre "johan/que_necesitas") para reproducirla luego sin transferir audio."""
        trozo = 1024
        datos = np.asarray(pcm16, dtype="<i2").tobytes()
        self._repl_listo()
        self.ser.write(f"parlante.guardar('/voces/{nombre}.pcm', {len(datos)}, {trozo})".encode() + b"\x04")
        self._hasta(b"LISTO\r\n", 5)
        for i in range(0, len(datos), trozo):
            self.ser.write(datos[i:i + trozo])
            self._hasta(b"K", 5)
        salida = self._hasta(b"\x04>", 10)
        if b"GUARDADO_OK" not in salida:
            raise RuntimeError(f"no se guardó {nombre} en la placa: {salida[-200:]!r}")

    def voces_en_placa(self) -> set[str]:
        """Nombres ("persona/clave") de las respuestas guardadas en la flash de la placa."""
        salida = self._ejecutar("parlante.listar_voces()", espera=10).removeprefix("OK")
        return {l.split()[1] for l in salida.splitlines() if l.startswith("VOZ ")}

    def reproducir_voz(self, nombre: str) -> None:
        """Reproduce una respuesta guardada con guardar_voz (sin transferir audio: < 0.5 s)."""
        self._ejecutar(f"parlante.reproducir_archivo('/voces/{nombre}.pcm')", espera=15)

    def leer(self, n_muestras: int) -> np.ndarray:
        necesito = n_muestras * 2
        datos = self._resto
        while len(datos) < necesito:
            trozo = self.ser.read(necesito - len(datos))
            if not trozo:
                raise TimeoutError("no llega audio por el cable")
            datos += trozo
        self._resto = datos[necesito:]
        return np.frombuffer(datos[:necesito], dtype="<i2").copy()

    def cerrar(self):
        import time
        if getattr(self, "_repl", False):
            # MicroPython: sale del REPL crudo y deja la placa esperando (sin reiniciarla), para que
            # el siguiente programa la encuentre lista. main.py (WiFi) vuelve al desconectar y
            # reconectar la placa.
            self.ser.write(b"\x02")
            time.sleep(0.1)
        elif getattr(self, "_en_modo_cable", False):
            self.ser.rts = True  # modo audio continuo: reiniciar para salir de él
            time.sleep(0.1)
            self.ser.rts = False
        self.ser.close()


class Control:
    """Órdenes a la ESP32: relés y audio para el parlante."""

    def __init__(self, host: str | None = None, timeout: float = 10.0):
        E = config()["esp32"]
        self.sock = socket.create_connection((_host(host), E["puerto_control"]), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.archivo = self.sock.makefile("rb")

    def _orden(self, d: dict, extra: bytes = b"") -> dict:
        self.sock.sendall(json.dumps(d).encode() + b"\n" + extra)
        linea = self.archivo.readline()
        if not linea:
            raise ConnectionError("la ESP32 cerró la conexión de control")
        return json.loads(linea)

    def ping(self) -> dict:
        return self._orden({"cmd": "ping"})

    def estado(self) -> dict:
        return self._orden({"cmd": "estado"})

    def rele(self, n: int, encendido: bool) -> dict:
        return self._orden({"cmd": "rele", "n": n, "on": encendido})

    def todos_off(self) -> dict:
        return self._orden({"cmd": "todos_off"})

    def reproducir(self, pcm16: np.ndarray) -> dict:
        datos = np.asarray(pcm16, dtype="<i2").tobytes()
        return self._orden({"cmd": "play", "bytes": len(datos)}, datos)

    def cerrar(self):
        self.sock.close()
