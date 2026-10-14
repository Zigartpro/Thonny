# Firmware de Thonny en MicroPython (ESP32-S3). Mismo protocolo que el firmware Arduino
# (ver PROTOCOLO.md), solo por WiFi:
#   puerto 5000: la ESP32 envía el audio del INMP441 (PCM 16 bits LE, 16 kHz, mono)
#   puerto 5001: órdenes JSON por línea: ping, rele, todos_off, play, estado
# Se sube con:  mpremote connect COM12 cp config.py secrets.py main.py :  + reset

import asyncio
import gc
import json
import time

import micropython
import network
from machine import I2S, Pin

import config as C
import secrets as S

# ------------------------------------------------------------------ relés (lo primero)
# Pin(..., value=...) fija el nivel de "apagado" antes de que el pin sea salida.
_APAGADO = 1 if C.RELE_ACTIVO_BAJO else 0
reles = [Pin(p, Pin.OUT, value=_APAGADO) for p in C.RELES]
estado_reles = [False, False, False, False]


def poner_rele(i, encendido):
    reles[i].value((1 - _APAGADO) if encendido else _APAGADO)
    estado_reles[i] = encendido


# ------------------------------------------------------------------ audio
mic = I2S(1, sck=Pin(C.MIC_SCK), ws=Pin(C.MIC_WS), sd=Pin(C.MIC_SD), mode=I2S.RX,
          bits=32, format=I2S.MONO, rate=C.SAMPLE_RATE, ibuf=16000)
amp = I2S(0, sck=Pin(C.AMP_BCLK), ws=Pin(C.AMP_LRC), sd=Pin(C.AMP_DIN), mode=I2S.TX,
          bits=16, format=I2S.MONO, rate=C.SAMPLE_RATE, ibuf=8000)


@micropython.viper
def a_16_bits(src: ptr32, dst: ptr16, n: int, shift: int):
    # 32 bits (24 útiles) -> 16 bits con saturación
    for i in range(n):
        v = src[i] >> shift
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        dst[i] = v


cliente_audio = None  # StreamWriter del PC conectado al puerto 5000


async def tarea_microfono():
    N = 512  # muestras por bloque (32 ms)
    crudo = bytearray(N * 4)
    pcm = bytearray(N * 2)
    lector = asyncio.StreamReader(mic)
    while True:
        n = await lector.readinto(crudo)
        w = cliente_audio
        if w is None or n <= 0:
            continue
        muestras = n // 4
        a_16_bits(crudo, pcm, muestras, C.MIC_SHIFT)
        try:
            w.write(memoryview(pcm)[:muestras * 2])
            await w.drain()
        except Exception:
            pass


async def atender_audio(reader, writer):
    global cliente_audio
    if cliente_audio is not None:
        try:
            cliente_audio.close()
        except Exception:
            pass
    cliente_audio = writer
    print("Audio: PC conectado")
    try:
        while True:  # solo para detectar cuándo se desconecta
            if not await reader.read(64):
                break
    except Exception:
        pass
    if cliente_audio is writer:
        cliente_audio = None
    print("Audio: PC desconectado")
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass


# ------------------------------------------------------------------ órdenes
sta = network.WLAN(network.STA_IF)


async def reproducir(reader, n):
    salida = asyncio.StreamWriter(amp)
    faltan = n
    while faltan > 0:
        trozo = await asyncio.wait_for(reader.readexactly(min(1024, faltan)), 3)
        salida.write(trozo)
        await salida.drain()
        faltan -= len(trozo)


async def atender_orden(reader, linea):
    try:
        d = json.loads(linea)
    except ValueError:
        return {"ok": False, "error": "json invalido"}
    cmd = d.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "rele":
        n, on = d.get("n"), d.get("on")
        if n not in (1, 2, 3, 4) or on not in (True, False):
            return {"ok": False, "error": "rele: n debe ser 1-4 y on true/false"}
        poner_rele(n - 1, on)
        print("Rele", n, "ON" if on else "OFF")
        return {"ok": True}
    if cmd == "todos_off":
        for i in range(4):
            poner_rele(i, False)
        return {"ok": True}
    if cmd == "play":
        n = d.get("bytes", -1)
        if not isinstance(n, int) or n <= 0 or n > C.MAX_PLAY_BYTES or n % 2:
            return {"ok": False, "error": "bytes invalido"}
        try:
            await reproducir(reader, n)
        except Exception:
            return {"ok": False, "error": "audio incompleto"}
        return {"ok": True}
    if cmd == "estado":
        gc.collect()
        return {"ok": True, "reles": [int(e) for e in estado_reles], "wifi": sta.isconnected(),
                "ip": sta.ifconfig()[0], "rssi": sta.status("rssi") if sta.isconnected() else 0,
                "heap": gc.mem_free(), "audio_conectado": cliente_audio is not None}
    return {"ok": False, "error": "comando desconocido"}


async def atender_control(reader, writer):
    print("Control: PC conectado")
    try:
        while True:
            linea = await reader.readline()
            if not linea:
                break
            respuesta = await atender_orden(reader, linea)
            writer.write(json.dumps(respuesta) + "\n")
            await writer.drain()
    except Exception as e:
        print("Control:", e)
    print("Control: PC desconectado")
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass


# ------------------------------------------------------------------ WiFi y arranque

async def conectar_wifi():
    network.hostname(C.NOMBRE_RED)
    sta.active(True)
    sta.config(txpower=C.WIFI_TXPOWER)  # evita el reinicio por caída de voltaje
    sta.config(pm=sta.PM_NONE)          # sin ahorro de energía: el audio fluye parejo
    while not sta.isconnected():
        print('Conectando a "%s"...' % S.WIFI_SSID)
        sta.connect(S.WIFI_SSID, S.WIFI_PASSWORD)
        for _ in range(40):
            if sta.isconnected():
                break
            await asyncio.sleep_ms(500)
        if not sta.isconnected():
            print("WiFi sin conexion, reintentando")
            sta.disconnect()
    print("WiFi OK. IP:", sta.ifconfig()[0], " RSSI:", sta.status("rssi"), "dBm")
    print("Tambien como %s.local" % C.NOMBRE_RED)


async def principal():
    print("\n== Thonny ESP32-S3 (MicroPython) ==")
    await conectar_wifi()
    await asyncio.start_server(atender_audio, "0.0.0.0", C.PUERTO_AUDIO)
    await asyncio.start_server(atender_control, "0.0.0.0", C.PUERTO_CONTROL)
    print("Audio en el puerto %d, control en el puerto %d" % (C.PUERTO_AUDIO, C.PUERTO_CONTROL))
    asyncio.create_task(tarea_microfono())
    while True:
        if not sta.isconnected():
            print("WiFi perdido")
            await conectar_wifi()
        await asyncio.sleep(5)


try:
    asyncio.run(principal())
finally:
    for i in range(4):
        poner_rele(i, False)
    mic.deinit()
    amp.deinit()
