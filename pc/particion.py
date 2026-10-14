"""Fase 3.5: partición fija train / val / test, y la regla de que test nunca se usa para entrenar.

Miembros (datos/clips.csv):
  - test: todo el Audio 5. Quien no tenga Audio 5 aporta a test sus últimas
    `sin_audio5.audio2_ultimas` activaciones del Audio 2 y los clips del Audio 3 desde
    `sin_audio5.audio3_desde` ("Hola Thonny, <comando>").
  - val: una fracción `val_fraccion` del resto de cada persona (para escoger el umbral).
  - train: lo demás.
Ajenos públicos (datos/ajenos.csv): se reparten HABLANTES completos, balanceados por sexo:
  `ajenos_test_por_sexo` a test (voces que el modelo nunca oye), `ajenos_val_por_sexo` a val
  y el resto a train. Los impostores sintéticos (datos/ajenos_sinteticos.csv) heredan la
  partición de su hablante original.

Uso:  python pc/particion.py        (reescribe la columna `particion` de los dos CSV)
Desde otros scripts:  from particion import cargar, afirmar_sin_test
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, ruta  # noqa: E402


def _leer(p: Path) -> tuple[list[dict], list[str]]:
    with open(p, encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f)
        return list(lector), lector.fieldnames


def _escribir(p: Path, filas: list[dict], columnas: list[str]) -> None:
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


def es_test_miembro(r: dict, con_audio5: set[str], P: dict) -> bool:
    if r["audio"] == "5":
        return True
    if r["hablante"] in con_audio5:
        return False
    if r["audio"] == "3":
        return int(r["n"]) >= P["sin_audio5"]["audio3_desde"]
    return False


def asignar() -> None:
    cfg = config()
    P = cfg["particion"]
    rng = random.Random(P["semilla"])

    clips, cols = _leer(ruta("clips_csv"))
    miembros = [r for r in clips if r["tipo"] == "miembro"]
    con_audio5 = {r["hablante"] for r in miembros if r["audio"] == "5"}
    for h in cfg["hablantes"]:
        propios = [r for r in miembros if r["hablante"] == h]
        # últimas activaciones del Audio 2 para quien no tiene Audio 5
        extra = set()
        if h not in con_audio5:
            a2 = sorted((r for r in propios if r["audio"] == "2"), key=lambda r: int(r["n"]))
            extra = {id(r) for r in a2[-P["sin_audio5"]["audio2_ultimas"]:]}
        resto = []
        for r in propios:
            if es_test_miembro(r, con_audio5, P) or id(r) in extra:
                r["particion"] = "test"
            else:
                resto.append(r)
        rng.shuffle(resto)
        n_val = round(len(resto) * P["val_fraccion"])
        for k, r in enumerate(resto):
            r["particion"] = "val" if k < n_val else "train"
    _escribir(ruta("clips_csv"), clips, cols)

    ajenos_path = ruta("datos") / "ajenos.csv"
    if ajenos_path.exists():
        ajenos, cols_a = _leer(ajenos_path)
        destino = {}
        for sexo in ("f", "m"):
            hablantes = sorted({r["hablante"] for r in ajenos if r["sexo"] == sexo})
            rng.shuffle(hablantes)
            n_t, n_v = P["ajenos_test_por_sexo"], P["ajenos_val_por_sexo"]
            destino |= {h: "test" for h in hablantes[:n_t]}
            destino |= {h: "val" for h in hablantes[n_t:n_t + n_v]}
        # los impostores sintéticos heredan la partición de su hablante original
        for p in (ajenos_path, ruta("datos") / "ajenos_sinteticos.csv"):
            if p.exists():
                filas_p, cols_p = _leer(p)
                for r in filas_p:
                    r["particion"] = destino.get(r["hablante"], "train")
                _escribir(p, filas_p, cols_p)

    resumen = Counter((f["origen"], f["particion"]) for f in cargar(("train", "val", "test")))
    for (origen, part), n in sorted(resumen.items()):
        print(f"  {origen:15s} {part:6s} {n}")


def cargar(particiones=("train",), version: str = "celular") -> list[dict]:
    """Filas unificadas: {etiqueta, hablante, origen, particion, ruta, audio, tipo_frase}.
    `etiqueta` es el nombre del miembro o "ajeno"."""
    filas = []
    clips, _ = _leer(ruta("clips_csv"))
    for r in clips:
        if r["particion"] in particiones and r["tipo"] == "miembro":
            rut = r["ruta_celular"] if version == "celular" else r["ruta_microfono"]
            if rut:
                filas.append({"etiqueta": r["hablante"], "hablante": r["hablante"], "origen": "miembro",
                              "particion": r["particion"], "ruta": str(RAIZ / rut), "audio": r["audio"],
                              "tipo_frase": r["tipo_frase"], "revisar": r["revisar"]})
    for p in (ruta("datos") / "ajenos.csv", ruta("datos") / "ajenos_sinteticos.csv"):
        if not p.exists():
            continue
        ajenos, _ = _leer(p)
        for r in ajenos:
            if r["particion"] in particiones:
                filas.append({"etiqueta": "ajeno", "hablante": r["hablante"], "origen": r["tipo"],
                              "particion": r["particion"], "ruta": r["ruta"], "audio": "",
                              "tipo_frase": "libre", "revisar": "0"})
    sin_asignar = [f for f in filas if f["particion"] not in ("train", "val", "test")]
    assert not sin_asignar, f"{len(sin_asignar)} clips sin partición: corre pc/particion.py"
    return filas


def afirmar_sin_test(filas: list[dict]) -> None:
    """Regla del proyecto: nada de test (ni ningún Audio 5) entra a entrenamiento o calibración."""
    malos = [f for f in filas if f["particion"] == "test" or (f["origen"] == "miembro" and f["audio"] == "5")]
    assert not malos, f"FUGA: {len(malos)} clips de test en entrenamiento/calibración, p. ej. {malos[0]['ruta']}"
    ajenos_test = {f["hablante"] for f in cargar(("test",)) if f["etiqueta"] == "ajeno"}
    usados = {f["hablante"] for f in filas if f["etiqueta"] == "ajeno"}
    assert not (ajenos_test & usados), f"FUGA: hablantes ajenos de test usados para entrenar: {ajenos_test & usados}"


if __name__ == "__main__":
    asignar()
