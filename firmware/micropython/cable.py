# Modo cable: manda el audio del INMP441 por el USB (UART0, chip CH343) en vez de WiFi.
# El PC lo arranca desde el REPL con `import cable`; la placa sube el UART a BAUDIOS_CABLE,
# escribe la marca b"THONNY_AUDIO\n" y desde ahí solo manda PCM 16 bits LE, 16 kHz, mono.
# Para salir, el PC reinicia la placa (pulso en RTS) y vuelve a arrancar main.py.

import time

import micropython
from machine import I2S, UART, Pin

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


def correr():
    mic = I2S(1, sck=Pin(C.MIC_SCK), ws=Pin(C.MIC_WS), sd=Pin(C.MIC_SD), mode=I2S.RX,
              bits=32, format=I2S.MONO, rate=C.SAMPLE_RATE, ibuf=16000)
    time.sleep_ms(100)  # deja salir la respuesta del REPL a la velocidad normal
    uart = UART(0, baudrate=C.BAUDIOS_CABLE)  # el UART del REPL (no admite cambiar el búfer)
    time.sleep_ms(50)
    uart.write(b"\nTHONNY_AUDIO\n")
    crudo = bytearray(512 * 4)
    pcm = bytearray(512 * 2)
    while True:
        n = mic.readinto(crudo) // 4
        _a_16_bits(crudo, pcm, n, C.MIC_SHIFT)
        uart.write(memoryview(pcm)[:n * 2])


correr()
