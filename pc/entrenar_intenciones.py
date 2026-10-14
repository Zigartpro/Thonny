"""Fase 2.3: entrena el clasificador de intenciones (TF-IDF + LogisticRegression).

  1. Validación cruzada estratificada sobre datos/intenciones.csv (exactitud por fold,
     reporte por clase y matriz de confusión).
  2. Prueba con texto real: las transcripciones de Whisper de los comandos de clips.csv
     (lo que de verdad le llegaría a Thonny), con la intención sacada de la frase esperada.
  3. Entrena con todo el dataset y guarda el modelo en modelos/intenciones.joblib.
Resultados en resultados/intenciones.json.

Uso:  python pc/entrenar_intenciones.py
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config, normalizar, ruta, texto_intencion  # noqa: E402


def crear_modelo(C: float) -> Pipeline:
    # n-gramas de palabras + n-gramas de caracteres (aguantan errores como "caudín" o "prancha")
    rasgos = FeatureUnion([
        ("palabras", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
        ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", rasgos), ("clf", LogisticRegression(C=C, max_iter=5000))])


def intencion_esperada(frase: str) -> str | None:
    """Intención de una frase del guion ("Apaga el cautín" -> apagar_cautin)."""
    t = normalizar(frase)
    accion = "prender" if any(v in t for v in ("prende", "enciende")) else \
             "apagar" if any(v in t for v in ("apaga", "desconecta")) else None
    aparato = next((a for k, a in (("luz", "bombillo"), ("bombillo", "bombillo"), ("ventilador", "ventilador"),
                                   ("cautin", "cautin"), ("plancha", "plancha")) if k in t), None)
    return f"{accion}_{aparato}" if accion and aparato else None


def main() -> None:
    cfg = config()
    I = cfg["intenciones"]
    with open(ruta("datos") / "intenciones.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    X = [texto_intencion(r["texto"]) for r in filas]
    y = np.array([r["intencion"] for r in filas])
    etiquetas = sorted(set(y))
    print(f"Dataset: {len(X)} frases, {len(etiquetas)} intenciones")

    # 1. Validación cruzada
    cv = StratifiedKFold(n_splits=I["folds"], shuffle=True, random_state=0)
    puntajes = cross_val_score(crear_modelo(I["C"]), X, y, cv=cv)
    pred_cv = cross_val_predict(crear_modelo(I["C"]), X, y, cv=cv)
    print(f"Validación cruzada ({I['folds']} folds): exactitud {puntajes.mean():.3f} ± {puntajes.std():.3f}")

    # 3. Modelo final con todo el dataset
    modelo = crear_modelo(I["C"]).fit(X, y)
    destino = RAIZ / I["modelo"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, destino)

    # 2. Prueba con transcripciones reales de Whisper
    with open(ruta("clips_csv"), encoding="utf-8") as f:
        comandos = [r for r in csv.DictReader(f) if r["tipo_frase"] == "comando"]
    reales, errores = [], []
    for r in comandos:
        esperada = intencion_esperada(r["frase_esperada"])
        texto = r["transcripcion_clip"]
        if esperada is None or not texto:
            continue
        proba = modelo.predict_proba([texto_intencion(texto)])[0]
        k = int(np.argmax(proba))
        pred, conf = modelo.classes_[k], float(proba[k])
        aceptada = conf >= I["umbral_confianza"]
        reales.append({"hablante": r["hablante"], "particion": r["particion"], "esperada": esperada,
                       "predicha": pred, "confianza": round(conf, 3), "aceptada": aceptada})
        if pred != esperada or not aceptada:
            errores.append({"clip": Path(r["ruta_celular"]).name, "transcripcion": texto,
                            "esperada": esperada, "predicha": pred, "confianza": round(conf, 3)})
    exactitud_real = float(np.mean([x["predicha"] == x["esperada"] for x in reales]))
    correcta_aceptada = float(np.mean([x["predicha"] == x["esperada"] and x["aceptada"] for x in reales]))
    print(f"Transcripciones reales: {len(reales)} comandos | exactitud {exactitud_real:.3f} | "
          f"correcta y sobre el umbral {I['umbral_confianza']}: {correcta_aceptada:.3f}")
    for e in errores:
        print(f"   {e['clip']:22s} '{e['transcripcion']}' -> {e['predicha']} ({e['confianza']}), esperada {e['esperada']}")
    por_hablante = {h: round(float(np.mean([x["predicha"] == x["esperada"] for x in reales if x["hablante"] == h])), 3)
                    for h in sorted({x["hablante"] for x in reales})}

    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"frases": len(X), "por_intencion": {e: int((y == e).sum()) for e in etiquetas}},
        "modelo": {"rasgos": "tfidf palabras (1-2) + caracteres char_wb (2-5)",
                   "clasificador": f"LogisticRegression C={I['C']}"},
        "validacion_cruzada": {
            "folds": I["folds"], "exactitud_por_fold": [round(float(p), 3) for p in puntajes],
            "exactitud_media": round(float(puntajes.mean()), 3), "desviacion": round(float(puntajes.std()), 3),
            "reporte": classification_report(y, pred_cv, output_dict=True, zero_division=0),
            "etiquetas": etiquetas, "matriz_confusion": confusion_matrix(y, pred_cv, labels=etiquetas).tolist(),
        },
        "transcripciones_reales": {
            "nota": "Comandos de clips.csv transcritos por Whisper. Algunos errores de Whisper de estos "
                    "mismos audios se usaron como vocabulario del dataset, así que es una prueba optimista.",
            "comandos": len(reales), "exactitud": round(exactitud_real, 3),
            "umbral_confianza": I["umbral_confianza"], "correcta_y_aceptada": round(correcta_aceptada, 3),
            "exactitud_por_hablante": por_hablante, "errores": errores,
        },
    }
    salida = ruta("resultados") / "intenciones.json"
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo -> {destino.relative_to(RAIZ)} | resultados -> {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
