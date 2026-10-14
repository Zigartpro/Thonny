"""Fase 4.3: evalúa el reconocimiento de hablante en los experimentos del plan.

  E1  línea base: un solo clip de "Hola Thonny" por persona (versión celular).
  E2  todos los clips de train (celular).
  E3  train celular + microfono (necesita los clips grabados por el INMP441; si no hay, se omite).
  E4  E2 + aumentos de datos (ruido, volumen y canal parlante->micrófono simulado).

Para cada experimento y clasificador (coseno, reglog): entrena con train, escoge el umbral con
val y mide en test: aceptación correcta por persona, identificación, falsa aceptación de
ajenos, matriz de confusión y barrido de umbral con EER. Guarda resultados/<exp>_<clf>.json,
una figura PNG y actualiza resultados/comparacion.md.

Uso:  python pc/evaluar.py [--experimento E2] [--clasificador coseno]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import config, ruta  # noqa: E402
from hablante import AJENO, CLASIFICADORES, Encoder  # noqa: E402
from particion import afirmar_sin_test, cargar  # noqa: E402

EXPERIMENTOS = {
    "E1": "Línea base: un clip de 'Hola Thonny' por persona (celular)",
    "E2": "Todos los clips de train (celular)",
    "E3": "Train celular + microfono",
    "E4": "E2 + aumentos (ruido, volumen, canal simulado)",
}
RECHAZADO = "rechazado"


# ------------------------------------------------------------------ aumentos

def aumentar(wav: np.ndarray, semilla: int) -> np.ndarray:
    from scipy.signal import butter, fftconvolve, sosfilt

    V = config()["verificacion"]
    rng = np.random.default_rng(semilla)
    x = wav.astype(np.float64)
    if rng.random() < 0.7:  # canal parlante -> micrófono: pasa-banda + reverberación corta
        lo, hi = V["canal_hz"]
        x = sosfilt(butter(4, [lo * rng.uniform(0.8, 1.5), hi * rng.uniform(0.7, 1.2)], "bandpass",
                           fs=16000, output="sos"), x)
        t = np.arange(int(16000 * rng.uniform(*V["reverb_s"]))) / 16000
        ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / t[-1])
        ir[0] = 1.0
        x = fftconvolve(x, ir / np.abs(ir).sum() * 4)[:len(x)]
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    snr = rng.uniform(*V["snr_db"])
    x = x + rng.standard_normal(len(x)) * rms / 10 ** (snr / 20)
    x = x * 10 ** (rng.uniform(*V["ganancia_db"]) / 20)
    return np.clip(x, -1, 1).astype(np.float32)


# ------------------------------------------------------------------ métricas

def correctos(etiquetas, pred) -> np.ndarray:
    return np.array([e == p for e, p in zip(etiquetas, pred)])


def curvas(etiquetas, pred, puntaje):
    """FAR y FRR para cada umbral. Un genuino cuenta como aceptado solo si además se identifica bien."""
    gen = np.array([e != AJENO for e in etiquetas])
    ok = correctos(etiquetas, pred)
    # umbrales candidatos: puntos medios entre puntajes vecinos (y los extremos), para que un
    # hueco sin errores quede representado por su centro y no por su borde
    u = np.unique(puntaje)
    umbrales = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    far = np.array([np.mean(puntaje[~gen] >= t) for t in umbrales])
    frr = np.array([1 - np.mean((puntaje[gen] >= t) & ok[gen]) for t in umbrales])
    return umbrales, far, frr


def punto_eer(umbrales, far, frr) -> tuple[float, float]:
    """Umbral donde FAR = FRR. Si hay un rango de umbrales empatados (p. ej. sin ningún error),
    se toma su punto medio en vez del borde."""
    dif = np.abs(far - frr)
    empatados = np.flatnonzero(np.isclose(dif, dif.min()))
    k = empatados[len(empatados) // 2]
    t = (umbrales[empatados[0]] + umbrales[empatados[-1]]) / 2
    return float(t), float((far[k] + frr[k]) / 2)


def escoger_umbral(etiquetas, pred, puntaje) -> float:
    V = config()["verificacion"]
    umbrales, far, frr = curvas(etiquetas, pred, puntaje)
    if V["criterio_umbral"] == "far":
        validos = umbrales[far <= V["far_objetivo"]]
        return float(validos.min())
    return punto_eer(umbrales, far, frr)[0]


def metricas(filas, pred, puntaje, umbral) -> dict:
    et = [f["etiqueta"] for f in filas]
    acept = puntaje >= umbral
    ok = correctos(et, pred)
    personas = config()["hablantes"]
    res = {"por_persona": {}, "far": {}}
    for p in personas:
        idx = np.array([e == p for e in et])
        act = idx & np.array([f["tipo_frase"] == "activacion" for f in filas])
        res["por_persona"][p] = {
            "clips": int(idx.sum()),
            "aceptacion_correcta": round(float(np.mean(acept[idx] & ok[idx])), 3),
            "aceptacion_correcta_activaciones": round(float(np.mean(acept[act] & ok[act])), 3) if act.any() else None,
            "identificacion": round(float(np.mean(ok[idx])), 3),
        }
    gen = np.array([e != AJENO for e in et])
    res["aceptacion_correcta_media"] = round(float(np.mean(acept[gen] & ok[gen])), 3)
    res["identificacion_media"] = round(float(np.mean(ok[gen])), 3)
    for origen in sorted({f["origen"] for f in filas if f["etiqueta"] == AJENO}):
        idx = np.array([f["etiqueta"] == AJENO and f["origen"] == origen for f in filas])
        res["far"][origen] = {"clips": int(idx.sum()), "falsa_aceptacion": round(float(np.mean(acept[idx])), 3)}
    filas_m = personas + [AJENO]
    cols = personas + [RECHAZADO]
    M = np.zeros((len(filas_m), len(cols)), dtype=int)
    for e, p, a in zip(et, pred, acept):
        M[filas_m.index(e), cols.index(p if a else RECHAZADO)] += 1
    res["matriz_confusion"] = {"filas": filas_m, "columnas": cols, "valores": M.tolist()}
    umbrales, far, frr = curvas(et, pred, puntaje)
    t_eer, eer = punto_eer(umbrales, far, frr)
    res["eer_test"] = round(eer, 3)
    res["umbral_eer_test"] = round(t_eer, 3)
    return res


# ------------------------------------------------------------------ figura y comparación

def figura(nombre, filas, pred, puntaje, umbral, res, destino: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    et = [f["etiqueta"] for f in filas]
    umbrales, far, frr = curvas(et, pred, puntaje)
    gen = np.array([e != AJENO for e in et])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(umbrales, far * 100, label="Falsa aceptación (ajenos)")
    a1.plot(umbrales, frr * 100, label="Falso rechazo (miembros)")
    a1.axvline(umbral, color="k", ls="--", label=f"umbral de val = {umbral:.2f}")
    a1.set_xlabel("umbral")
    a1.set_ylabel("%")
    a1.set_title(f"{nombre}: barrido de umbral (EER test {res['eer_test'] * 100:.1f} %)")
    a1.legend(fontsize=8)
    a2.hist(puntaje[gen], bins=30, alpha=0.6, label="miembros")
    a2.hist(puntaje[~gen], bins=30, alpha=0.6, label="ajenos")
    a2.axvline(umbral, color="k", ls="--")
    a2.set_xlabel("puntaje")
    a2.set_title("Distribución de puntajes en test")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destino, dpi=110)
    plt.close(fig)


def actualizar_comparacion():
    filas = []
    import re
    for p in sorted(q for q in ruta("resultados").glob("*.json") if re.fullmatch(r"E\d_\w+\.json", q.name)):
        d = json.loads(p.read_text(encoding="utf-8"))
        t = d["test"]
        far = " / ".join(f"{o}: {v['falsa_aceptacion'] * 100:.1f} %" for o, v in t["far"].items())
        filas.append(f"| {d['experimento']} | {d['clasificador']} | {d['train']['miembros']} | {d['umbral']:.3f} | "
                     f"{t['aceptacion_correcta_media'] * 100:.1f} % | {t['identificacion_media'] * 100:.1f} % | "
                     f"{far} | {t['eer_test'] * 100:.1f} % |")
    texto = ("# Comparación de experimentos de reconocimiento de hablante\n\n"
             "Generado por `pc/evaluar.py`. Umbral escogido en `val`; todas las métricas sobre `test` "
             "(Audio 5 de cristian y johan, clips apartados de rocko, y 6 hablantes públicos que el modelo nunca vio).\n\n"
             "| Exp. | Clasificador | Clips de miembros en train | Umbral | Aceptación correcta (miembros) | "
             "Identificación | Falsa aceptación (ajenos) | EER test |\n|---|---|---|---|---|---|---|---|\n"
             + "\n".join(filas) + "\n")
    (ruta("resultados") / "comparacion.md").write_text(texto, encoding="utf-8")


# ------------------------------------------------------------------ principal

def conjunto_train(exp: str) -> list[dict] | None:
    train = cargar(("train",))
    if exp == "E1":
        base = []
        for p in config()["hablantes"]:
            act = sorted((f for f in train if f["etiqueta"] == p and f["tipo_frase"] == "activacion"
                          and f["revisar"] == "0"), key=lambda f: (f["audio"], f["ruta"]))
            base.append(act[0])
        return base + [f for f in train if f["etiqueta"] == AJENO]
    if exp == "E3":
        mic = [f for f in cargar(("train",), version="microfono") if f["origen"] == "miembro"]
        return train + mic if mic else None
    return train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experimento", action="append", choices=list(EXPERIMENTOS))
    ap.add_argument("--clasificador", action="append", choices=list(CLASIFICADORES))
    args = ap.parse_args()
    V = config()["verificacion"]
    exps = args.experimento or list(EXPERIMENTOS)
    clfs = args.clasificador or V["clasificadores"]

    enc = Encoder()
    val, test = cargar(("val",)), cargar(("test",))
    afirmar_sin_test(val)  # el umbral tampoco se calibra con test
    X_val, X_test = enc.matriz(val), enc.matriz(test)

    for exp in exps:
        train = conjunto_train(exp)
        if train is None:
            print(f"{exp}: omitido ({EXPERIMENTOS[exp]}; no hay clips de micrófono todavía)")
            continue
        afirmar_sin_test(train)
        X = enc.matriz(train)
        y = [f["etiqueta"] for f in train]
        if exp == "E4":
            for k in range(V["n_aumentos"]):
                X = np.vstack([X, enc.matriz(train, variante=f"aum{k}", transformar=aumentar)])
                y = y + [f["etiqueta"] for f in train]
        enc.guardar()
        for c in clfs:
            modelo = CLASIFICADORES[c]().fit(X, y)
            pv, sv = modelo.puntuar(X_val)
            umbral = escoger_umbral([f["etiqueta"] for f in val], pv, sv)
            pt, st = modelo.puntuar(X_test)
            res = metricas(test, pt, st, umbral)
            nombre = f"{exp}_{c}"
            figura(nombre, test, pt, st, umbral, res, ruta("resultados") / f"{nombre}.png")
            salida = {
                "fecha": datetime.now().isoformat(timespec="seconds"), "experimento": exp,
                "descripcion": EXPERIMENTOS[exp], "clasificador": c,
                "train": {"miembros": sum(e != AJENO for e in y), "ajenos": sum(e == AJENO for e in y)},
                "criterio_umbral": V["criterio_umbral"], "umbral": round(umbral, 4),
                "val": {"clips": len(val)}, "test": res,
                "nota": "Solo versión celular: aún no hay clips grabados por el INMP441 (versión microfono).",
            }
            (ruta("resultados") / f"{nombre}.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
            pp = " ".join(f"{p}={v['aceptacion_correcta'] * 100:.0f}%" for p, v in res["por_persona"].items())
            far = " ".join(f"{o}={v['falsa_aceptacion'] * 100:.1f}%" for o, v in res["far"].items())
            print(f"{nombre:10s} train={len(y):5d} umbral={umbral:.3f} | aceptación {pp} | "
                  f"identif. {res['identificacion_media'] * 100:.0f}% | FAR {far} | EER test {res['eer_test'] * 100:.1f}%")
    actualizar_comparacion()
    print(f"\nTabla en {(ruta('resultados') / 'comparacion.md')}")


if __name__ == "__main__":
    main()
