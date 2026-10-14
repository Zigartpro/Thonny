# Grabación por cable: graba en la RAM de la placa y después la envía por el REPL (115200).
# No es tiempo real (transferir tarda ~3 veces lo grabado) pero la calidad es la misma que
# por WiFi. El PC lo usa así, desde el REPL crudo:
#     import grabar
#     grabar.grabar(3000)      # ms; responde b"THONNY_PCM <n>\n" + n bytes de PCM 16 bits LE
# Máximo 3 s por llamada (RAM de la placa, sin PSRAM).
#
# El micrófono y los bloques de RAM se reservan una sola vez al importar y se reutilizan:
# crearlos en cada grabación fragmentaba la memoria y la segunda fallaba (ESP_ERR_NO_MEM).

import sys

import micropython
from machine import I2S, Pin

import config as C


@micropython.viper
def _a_16_bits(src: ptr32, dst: ptr16, n: int, shift: int):
    for i in range(n):
        v = src[i] >> shift
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        dst[i] = v


BLOQUE = 8000          # muestras por bloque (0.5 s)
MAX_MS = 3000
_crudo = bytearray(512 * 4)
_bloques = [bytearray(BLOQUE * 2) for _ in range(C.SAMPLE_RATE * MAX_MS // 1000 // BLOQUE)]
_mic = I2S(1, sck=Pin(C.MIC_SCK), ws=Pin(C.MIC_WS), sd=Pin(C.MIC_SD), mode=I2S.RX,
           bits=32, format=I2S.MONO, rate=C.SAMPLE_RATE, ibuf=16000)


def grabar(ms, shift=C.MIC_SHIFT):
    n = C.SAMPLE_RATE * min(ms, MAX_MS) // 1000
    # vacía lo acumulado y descarta ~0.6 s: el INMP441 tarda en asentar su continua
    for _ in range(C.SAMPLE_RATE * 6 // 10 // 512 + 8):
        _mic.readinto(_crudo)
    pos = 0
    for b in _bloques:
        if pos >= n:
            break
        destino = memoryview(b)
        m = min(BLOQUE, n - pos)
        k = 0
        while k < m:
            leidas = _mic.readinto(memoryview(_crudo)[:min(512, m - k) * 4]) // 4
            _a_16_bits(_crudo, destino[k * 2:], leidas, shift)
            k += leidas
        pos += m
    sys.stdout.buffer.write(b"THONNY_PCM %d\n" % (n * 2))
    pos = 0
    for b in _bloques:
        if pos >= n:
            break
        m = min(BLOQUE, n - pos)
        sys.stdout.buffer.write(memoryview(b)[:m * 2])
        pos += m
