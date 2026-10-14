# Parlante por cable: el PC deja el audio en la RAM de la placa (en base64, por el REPL) y
# luego se reproduce por el MAX98357. También genera un tono de prueba sin transferir nada.
#     import parlante
#     parlante.tono(440, 1000)          # Hz, ms
#     parlante.reproducir(buf)          # buf: PCM 16 bits LE, 16 kHz, mono (máximo ~4 s)

import math
import struct

from machine import I2S, Pin

import config as C


def _amp():
    return I2S(0, sck=Pin(C.AMP_BCLK), ws=Pin(C.AMP_LRC), sd=Pin(C.AMP_DIN), mode=I2S.TX,
               bits=16, format=I2S.MONO, rate=C.SAMPLE_RATE, ibuf=8000)


def _terminar(amp):
    amp.write(bytearray(8000))  # silencio para que suene el final antes de apagar el I2S
    amp.deinit()


def tono(hz=440, ms=1000, volumen=0.3):
    periodo = C.SAMPLE_RATE // hz
    ciclo = bytearray(periodo * 2)
    for i in range(periodo):
        struct.pack_into("<h", ciclo, i * 2, int(volumen * 32767 * math.sin(2 * math.pi * i / periodo)))
    amp = _amp()
    try:
        for _ in range(C.SAMPLE_RATE * ms // 1000 // periodo):
            amp.write(ciclo)
    finally:
        _terminar(amp)
    print("TONO_OK")


def reproducir(buf):
    amp = _amp()
    try:
        amp.write(buf)
    finally:
        _terminar(amp)
    print("PLAY_OK")


def recibir(n_bytes, trozo=1024):
    # Recibe n_bytes de PCM crudo por el cable directo en los bloques de grabar.py (~4 veces
    # más rápido que mandarlo en base64). Responde "K" tras cada trozo para que el PC no llene
    # el búfer de entrada. Ctrl-C se desactiva: el byte 0x03 puede venir dentro del audio.
    import sys
    import grabar
    import micropython
    micropython.kbd_intr(-1)
    try:
        sys.stdout.write("LISTO\n")
        pos = 0
        for b in grabar._bloques:
            if pos >= n_bytes:
                break
            mv = memoryview(b)
            m = min(len(b), n_bytes - pos)
            k = 0
            while k < m:
                fin = k + min(trozo, m - k)
                while k < fin:
                    d = sys.stdin.buffer.read(fin - k)
                    mv[k:k + len(d)] = d
                    k += len(d)
                sys.stdout.write("K")
            pos += m
    finally:
        micropython.kbd_intr(3)


def guardar(ruta, n_bytes, trozo=1024):
    # Igual que recibir(), pero guarda el PCM en la flash (p. ej. /voces/johan/que_necesitas.pcm)
    # para reproducirlo después sin transferir nada (pc/subir_voces.py).
    import os
    import sys
    import micropython
    carpeta = ""
    for parte in ruta.split("/")[1:-1]:
        carpeta += "/" + parte
        try:
            os.mkdir(carpeta)
        except OSError:
            pass
    buf = bytearray(trozo)
    micropython.kbd_intr(-1)
    try:
        sys.stdout.write("LISTO\n")
        with open(ruta + ".tmp", "wb") as f:
            k = 0
            while k < n_bytes:
                m = min(trozo, n_bytes - k)
                mv = memoryview(buf)[:m]
                j = 0
                while j < m:
                    d = sys.stdin.buffer.read(m - j)
                    mv[j:j + len(d)] = d
                    j += len(d)
                f.write(mv)
                k += m
                sys.stdout.write("K")
    finally:
        micropython.kbd_intr(3)
    try:
        os.remove(ruta)
    except OSError:
        pass
    os.rename(ruta + ".tmp", ruta)
    print("GUARDADO_OK")


def reproducir_archivo(ruta):
    # Reproduce un PCM guardado en la flash, por trozos (no necesita RAM extra).
    import grabar
    buf = grabar._bloques[0]
    amp = _amp()
    try:
        with open(ruta, "rb") as f:
            while True:
                n = f.readinto(buf)
                if not n:
                    break
                amp.write(memoryview(buf)[:n])
    finally:
        _terminar(amp)
    print("PLAY_OK")


def listar_voces():
    import os
    try:
        personas = os.listdir("/voces")
    except OSError:
        personas = []
    for p in personas:
        for a in os.listdir("/voces/" + p):
            if a.endswith(".pcm"):
                print("VOZ", p + "/" + a[:-4])


def reproducir_bloques(n_bytes):
    # Reproduce lo que el PC dejó en los bloques ya reservados de grabar.py (máximo 3 s).
    # Así no se pide RAM nueva: un bytearray de ~45 KB fallaba por memoria fragmentada.
    import grabar
    amp = _amp()
    try:
        pos = 0
        for b in grabar._bloques:
            if pos >= n_bytes:
                break
            m = min(len(b), n_bytes - pos)
            amp.write(memoryview(b)[:m])
            pos += m
    finally:
        _terminar(amp)
    print("PLAY_OK")
