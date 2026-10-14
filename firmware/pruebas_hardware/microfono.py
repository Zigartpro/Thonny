# Prueba del micrófono INMP441 en ESP32-S3 (MicroPython)
# Muestra en la consola una barra con el nivel de sonido.
#
# Conexiones:
#   INMP441   ->  ESP32-S3
#   VDD       ->  3V3   (¡NO a 5V!)
#   GND       ->  GND
#   L/R       ->  GND
#   SCK       ->  GPIO 15
#   WS        ->  GPIO 16
#   SD        ->  GPIO 17

from machine import I2S, Pin
import array
import math
import time

SCK_PIN = 15
WS_PIN = 16
SD_PIN = 17

mic = I2S(
    1,
    sck=Pin(SCK_PIN),
    ws=Pin(WS_PIN),
    sd=Pin(SD_PIN),
    mode=I2S.RX,
    bits=32,
    format=I2S.STEREO,  # se leen ambos canales y se usa el que tenga señal
    rate=16000,
    ibuf=16000,
)

muestras = array.array("i", [0] * 512)  # 256 pares izquierda/derecha
MAXIMO = 8388607  # valor máximo de 24 bits
ANCHO = 40

print("Habla o aplaude cerca del micrófono (Ctrl+C para parar)")
time.sleep(0.5)

try:
    while True:
        n = mic.readinto(muestras) // 4
        pico_izq = 0
        pico_der = 0
        for i in range(0, n - 1, 8):
            izq = abs(muestras[i] >> 8)
            der = abs(muestras[i + 1] >> 8)
            if izq > pico_izq:
                pico_izq = izq
            if der > pico_der:
                pico_der = der

        pico = max(pico_izq, pico_der)
        canal = "IZQ" if pico_izq >= pico_der else "DER"

        if pico == 0:
            print("Sin señal: revisa VDD, GND, SD y SCK")
            time.sleep(0.5)
            continue

        db = 20 * math.log10(pico / MAXIMO)        # 0 dB = máximo
        largo = int((db + 60) / 60 * ANCHO)          # -60..0 dB -> 0..40
        largo = max(0, min(ANCHO, largo))
        print("{} {:6.1f} dB |{}{}|".format(canal, db, "#" * largo, " " * (ANCHO - largo)))
        time.sleep_ms(80)
except KeyboardInterrupt:
    pass
finally:
    mic.deinit()
    print("Fin")
             