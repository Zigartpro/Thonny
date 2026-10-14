"""Utilidades compartidas: configuración y normalización de texto."""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml
from rapidfuzz import fuzz

RAIZ = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def config() -> dict:
    with open(RAIZ / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(clave: str) -> Path:
    return RAIZ / config()["rutas"][clave]


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sin puntuación, espacios simples."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9ñ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def canonica(palabra: str) -> str:
    """Palabra normalizada; las variantes de "Thonny" y "hola" quedan unificadas."""
    act = config()["activacion"]
    p = normalizar(palabra)
    if p in act["variantes"] or fuzz.ratio(p, "thonny") >= act["similitud_variante"]:
        return "thonny"
    if fuzz.ratio(p, "hola") >= 80:  # "ola" sí, "hora" no
        return "hola"
    return p


def tokens(texto: str) -> list[str]:
    return [canonica(p) for p in normalizar(texto).split()]


def texto_intencion(texto: str) -> str:
    """Texto listo para el clasificador de intenciones: normalizado y sin la palabra "Thonny"."""
    return " ".join(t for t in tokens(texto) if t != "thonny")


def similitud(a: str, b: str) -> float:
    """Similitud 0-100 entre dos textos, después de unificar variantes."""
    return fuzz.ratio(" ".join(tokens(a)), " ".join(tokens(b)))
