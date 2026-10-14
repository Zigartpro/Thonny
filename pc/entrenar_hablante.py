"""Fase 4.5: entrena el modelo de hablante que usa Thonny y lo guarda en modelos/hablante.joblib.

Usa el experimento y el clasificador de `verificacion.modelo_final` en config.yaml (por
defecto E4 + reglog: el que mejor aguanta el cambio de micrófono). Entrena con train, escoge
el umbral con val (nunca con test) y guarda {modelo, umbral, personas, experimento}.

Uso:  python pc/entrenar_hablante.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, ruta  # noqa: E402
from evaluar import aumentar, conjunto_train, escoger_umbral  # noqa: E402
from hablante import AJENO, CLASIFICADORES, Encoder  # noqa: E402
from particion import afirmar_sin_test, cargar  # noqa: E402


def main() -> None:
    V = config()["verificacion"]
    exp, clf = V["modelo_final"]["experimento"], V["modelo_final"]["clasificador"]
    enc = Encoder()
    train = conjunto_train(exp)
    afirmar_sin_test(train)
    X = enc.matriz(train)
    y = [f["etiqueta"] for f in train]
    if exp == "E4":
        for k in range(V["n_aumentos"]):
            X = np.vstack([X, enc.matriz(train, variante=f"aum{k}", transformar=aumentar)])
            y = y + [f["etiqueta"] for f in train]
    val = cargar(("val",))
    afirmar_sin_test(val)
    enc.guardar()

    modelo = CLASIFICADORES[clf]().fit(X, y)
    pv, sv = modelo.puntuar(enc.matriz(val))
    umbral = escoger_umbral([f["etiqueta"] for f in val], pv, sv)

    destino = RAIZ / V["modelo_final"]["archivo"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "umbral": float(umbral), "personas": modelo.personas,
                 "experimento": exp, "clasificador": clf}, destino)
    resumen = {"fecha": datetime.now().isoformat(timespec="seconds"), "experimento": exp,
               "clasificador": clf, "umbral_val": round(float(umbral), 4),
               "clips_train": {"miembros": sum(e != AJENO for e in y), "ajenos": sum(e == AJENO for e in y)}}
    (ruta("resultados") / "modelo_hablante.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    print(f"Modelo {exp}/{clf} -> {destino.relative_to(RAIZ)} | umbral (val) {umbral:.3f} | personas {modelo.personas}")


if __name__ == "__main__":
    main()
