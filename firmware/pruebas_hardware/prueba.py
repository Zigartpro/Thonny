# Prueba del amplificador MAX98357 en ESP32-S3 (MicroPython)
# Reproduce una escala de tonos por I2S.
#
# Conexiones:
#   MAX98357   ->  ESP32-S3
#   VIN        ->  5V (o 3V3, suena más bajo)
#   GND        ->  GND
#   BCLK       ->  GPIO 5
#   LRC        ->  GPIO 6
#   DIN        ->  GPIO 7
#   SD         ->  sin conectar (o a 3V3)
#   GAIN       ->  sin conectar (9 dB)
#   Altavoz (4-8 ohm) entre + y -

from machine import I2S, Pin
import array
import math
import time

BCLK_PIN = 5
LRC_PIN = 6
DIN_PIN = 7

RATE = 16000      # muestras por segundo
VOLUMEN = 0.3     # 0.0 a 1.0

audio = I2S(
    0,
    sck=Pin(BCLK_PIN),
    ws=Pin(LRC_PIN),
    sd=Pin(DIN_PIN),
    mode=I2S.TX,
    bits=16,
    format=I2S.MONO,
    rate=RATE,
    ibuf=8000,
)


def generar_tono(frecuencia):
    # Un ciclo completo de seno en enteros de 16 bits
    muestras = RATE // frecuencia
    amp = int(32767 * VOLUMEN)
    buf = array.array("h", [0] * muestras)
    for i in range(muestras):
        buf[i] = int(amp * math.sin(2 * math.pi * i / muestras))
    return buf


def tocar(frecuencia, segundos):
    buf = generar_tono(frecuencia)
    fin = time.ticks_add(time.ticks_ms(), int(segundos * 1000))
    while time.ticks_diff(fin, time.ticks_ms()) > 0:
        audio.write(buf)


# Escala Do-Re-Mi-Fa-Sol-La-Si-Do
NOTAS = [262, 294, 330, 349, 392, 440, 494, 523]

try:
    print("Reproduciendo escala...")
    for f in NOTAS:
        print(f, "Hz")
        tocar(f, 0.4)
    print("Tono continuo de 440 Hz (Ctrl+C para parar)")
    while True:
        tocar(440, 1)
except KeyboardInterrupt:
    pass
finally:
    audio.deinit()
    print("Fin")
