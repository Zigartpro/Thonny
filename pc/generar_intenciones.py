"""Fase 2.3: arma datos/intenciones.csv (texto, intencion).

Las 8 intenciones de aparatos salen de combinar verbos x nombres del aparato x plantillas,
incluyendo errores típicos de Whisper vistos al transcribir los audios del proyecto
("caudín", "cotino", "prenda", "apague", "a parar"…). Se toman hasta `por_intencion` frases
distintas por intención con semilla fija. "saludo" y "desconocido" están escritas a mano.

Uso:  python pc/generar_intenciones.py
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, normalizar, ruta  # noqa: E402

POR_INTENCION = 30
SEMILLA = 7

PRENDER = ["prende", "enciende", "activa", "pon", "prendé", "préndeme",
           # errores de Whisper vistos en los logs
           "prenda", "encienda", "enciendo", "prended", "prender"]
APAGAR = ["apaga", "desconecta", "quita", "apágame", "ya apaga",
          # errores de Whisper vistos en los logs
          "apague", "desconecte", "apagar", "a parar"]
INFINITIVO = {"prender": ["prender", "encender", "activar", "poner"],
              "apagar": ["apagar", "desconectar", "quitar"]}

APARATOS = {
    "bombillo": ["la luz", "el bombillo", "el foco", "la lámpara", "la luz del cuarto", "el bombillo del cuarto"],
    "ventilador": ["el ventilador", "el abanico", "el ventilador del cuarto", "el bentilador"],
    "cautin": ["el cautín", "el cautin", "el soldador",
               # errores de Whisper vistos en los logs
               "el caudín", "el caoutín", "el cotino", "el cortín", "el cotín", "el cut-in"],
    "plancha": ["la plancha", "la plancha de cabello", "la plancha del pelo", "la planchita",
                # errores de Whisper vistos en los logs
                "la plancha de cabeza", "la prancha"],
}

PLANTILLAS = ["{v} {a}", "{v} {a} por favor", "por favor {v} {a}", "hola {v} {a}",
              "{v} {a} ya", "oye {v} {a}", "hola thonny {v} {a}", "tommy {v} {a}"]
PLANTILLAS_INF = ["puedes {i} {a}", "quiero {i} {a}", "me ayudas a {i} {a}", "necesito {i} {a}"]

SALUDO = [
    "hola", "hola thonny", "buenos días", "buenas tardes", "buenas noches", "cómo estás",
    "hola cómo estás", "qué más", "quiubo", "hola qué tal", "qué tal", "saludos", "hola buenas",
    "buenas", "hola cómo vas", "cómo vas", "qué hubo", "hola hola", "buen día", "hola amigo",
    "cómo amaneciste", "hola tommy", "qué más pues", "cómo te va", "hola de nuevo",
]
DESCONOCIDO = [
    "qué hora es", "cuánto es dos más dos", "cuéntame un chiste", "pon música", "sube el volumen",
    "abre la puerta", "cierra la ventana", "llama a mi mamá", "qué día es hoy", "va a llover mañana",
    "el ventilador es nuevo", "dónde está la plancha", "la luz está muy bonita", "compré un cautín",
    "apaga eso", "prende eso", "no sé", "gracias", "nada", "olvídalo", "mañana tengo clase temprano",
    "el perro del vecino ladró toda la noche", "me gusta el café con pan", "necesito comprar arroz",
    "ese partido estuvo muy bueno", "gracias por ver el video", "suscríbete al canal",
    "cuál es tu nombre", "cuántos años tienes", "qué puedes hacer",
]


def frases_aparato(accion: str, aparato: str) -> list[str]:
    verbos = PRENDER if accion == "prender" else APAGAR
    out = {p.format(v=v, a=a) for p in PLANTILLAS for v in verbos for a in APARATOS[aparato]}
    out |= {p.format(i=i, a=a) for p in PLANTILLAS_INF for i in INFINITIVO[accion] for a in APARATOS[aparato]}
    return sorted(out)


def main() -> None:
    rng = random.Random(SEMILLA)
    filas = []
    for aparato in APARATOS:
        for accion in ("prender", "apagar"):
            frases = frases_aparato(accion, aparato)
            filas += [(f, f"{accion}_{aparato}") for f in rng.sample(frases, min(POR_INTENCION, len(frases)))]
    filas += [(f, "saludo") for f in SALUDO]
    filas += [(f, "desconocido") for f in DESCONOCIDO]

    vistas = {}
    for texto, intencion in filas:
        clave = normalizar(texto)
        assert vistas.setdefault(clave, intencion) == intencion, f"'{texto}' aparece con dos intenciones"

    destino = ruta("datos") / "intenciones.csv"
    with open(destino, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["texto", "intencion"])
        w.writerows(filas)
    conteo = {}
    for _, i in filas:
        conteo[i] = conteo.get(i, 0) + 1
    print(f"{len(filas)} frases -> {destino.relative_to(RAIZ)}")
    for i, n in conteo.items():
        print(f"  {i:20s} {n}")


if __name__ == "__main__":
    main()
