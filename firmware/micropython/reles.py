# Relés por cable (desde el REPL):  import reles;  reles.poner(1, True);  reles.todos_off()
# Al importar, los 4 quedan en "apagado" (Pin(..., value=...) fija el nivel antes de ser salida).

from machine import Pin

import config as C

_APAGADO = 1 if C.RELE_ACTIVO_BAJO else 0
_pines = [Pin(p, Pin.OUT, value=_APAGADO) for p in C.RELES]
estado = [False, False, False, False]


def poner(n, encendido):
    _pines[n - 1].value((1 - _APAGADO) if encendido else _APAGADO)
    estado[n - 1] = bool(encendido)
    print("RELE_OK", n, int(encendido))


def todos_off():
    for n in range(1, 5):
        poner(n, False)
