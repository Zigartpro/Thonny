# Prueba de eco: INMP441 (micrófono) -> MAX98357 (altavoz) en ESP32-S3
# Lo que entra por el micrófono sale por el altavoz.
#
# Conexiones (las mismas de prueba.py y microfono.py):
#   MAX98357: VIN->5V, GND->GND, BCLK->5, LRC->6, DIN->7
#   INMP441:  VDD->3V3, GND->GND, L/R->GND, SCK->15, WS->16, SD->17
#
# Aleja el micrófono del altavoz para evitar el pitido (acople).

from machine import I2S, Pin

RATE = 16000
GANANCIA = 1   # 0 = sin amplificar, cada +1 duplica el volumen (no pases de 2)

mic = I2S(
    1,
    sck=Pin(15),
    ws=Pin(16),
    sd=Pin(17),
    mode=I2S.RX,
    bits=32,
    format=I2S.MONO,
    rate=RATE,
    ibuf=8000,
)

parlante = I2S(
    0,
    sck=Pin(5),
    ws=Pin(6),
    sd=Pin(7),
    mode=I2S.TX,
    bits=32,
    format=I2S.MONO,
    rate=RATE,
    ibuf=8000,
)

buf = bytearray(1024)  # 256 muestras mono de 32 bits (~16 ms)

print("Eco activo: habla al micrófono (Ctrl+C para parar)")

try:
    while True:
        n = mic.readinto(buf)
        if GANANCIA:
            I2S.shift(buf=buf, bits=32, shift=GANANCIA)
        parlante.write(buf[:n])
except KeyboardInterrupt:
    pass
finally:
    mic.deinit()
    parlante.deinit()
    print("Fin")
