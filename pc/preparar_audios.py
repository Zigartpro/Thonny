"""Fase 3.1: convierte los audios crudos en clips de una frase y llena datos/clips.csv.

Para cada audio de datos/crudos/<hablante>/audio<N>.* (y datos/crudos/ajenos/<nombre>/):
  1. Lo convierte a 16 kHz mono con ffmpeg.
  2. Lo corta en tramos por silencios (pydub) y transcribe cada tramo con faster-whisper,
     con marcas de tiempo por palabra. Se descartan las alucinaciones típicas de Whisper.
  3. Empareja lo oído con lo esperado (guion_audios.yaml):
       - "silencio": si hay tantos tramos como frases esperadas y el texto cuadra en promedio.
       - "alineacion": si no, alinea las palabras reconocidas con las del guion y corta entre
         palabras (sirve cuando la persona no dejó pausas).
       - "ocurrencias": en los audios de "Hola Thonny" repetido, extrae todas las que haya.
  4. Guarda cada clip normalizado en datos/clips/celular/<hablante>/<hablante>_a<N>_<NN>.wav.
Los clips dudosos quedan con revisar=1 en clips.csv; el resumen va a resultados/preparar_audios.json.

Uso:  python pc/preparar_audios.py [--hablante johan] [--audio 2] [--sin-escribir]
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_dlls  # noqa: E402,F401  (antes de faster_whisper)
from comun import RAIZ, canonica, config, normalizar, ruta, similitud, tokens  # noqa: E402

COLUMNAS = ["hablante", "tipo", "audio", "n", "tipo_frase", "frase_esperada", "duracion_s",
            "ruta_celular", "ruta_microfono", "particion", "transcripcion", "similitud",
            "transcripcion_clip", "similitud_clip", "revisar", "metodo", "inicio_s", "fin_s"]
PATRON_AUDIO = re.compile(r"^audio(\d+)\.\w+$")
HABLA_LIBRE = "(habla libre)"


@dataclass
class Palabra:
    texto: str
    canon: str
    ini: float
    fin: float


@dataclass
class Unidad:
    tipo_frase: str
    texto: str


@dataclass
class Clip:
    tipo_frase: str
    frase_esperada: str
    ini: float
    fin: float
    cobertura: float = 1.0
    parcial: bool = False


# ---------------------------------------------------------------- audio

def cargar(archivo: Path, sr: int) -> np.ndarray:
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(archivo), "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32).copy()


def tramos_por_silencio(x: np.ndarray, sr: int, P: dict) -> list[tuple[float, float]]:
    seg = AudioSegment((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes(),
                       sample_width=2, frame_rate=sr, channels=1)
    t = detect_nonsilent(seg, min_silence_len=P["silencio_min_ms"],
                         silence_thresh=seg.dBFS + P["umbral_silencio_db"])
    return [(s / 1000, e / 1000) for s, e in t]


def subdividir(x: np.ndarray, sr: int, tramos, P: dict) -> list[tuple[float, float]]:
    """Parte los tramos largos ("hola thonny hola thonny…" sin pausas) en ventanas de hasta
    `subdividir_s`, cortando solo en micro-pausas. Whisper se salta repeticiones en tramos
    largos, pero con 1-2 repeticiones por ventana las transcribe bien."""
    out = []
    for s, e in tramos:
        if e - s <= P["subdividir_s"]:
            out.append((s, e))
            continue
        trozo = x[int(s * sr):int(e * sr)]
        seg = AudioSegment((np.clip(trozo, -1, 1) * 32767).astype(np.int16).tobytes(),
                           sample_width=2, frame_rate=sr, channels=1)
        partes = detect_nonsilent(seg, min_silence_len=P["sub_silencio_ms"],
                                  silence_thresh=seg.dBFS + P["sub_umbral_db"])
        if not partes:
            out.append((s, e))
            continue
        ini, fin = partes[0]
        for a, b in partes[1:]:
            if (b - ini) / 1000 <= P["subdividir_s"]:
                fin = b
            else:
                out.append((s + ini / 1000, s + fin / 1000))
                ini, fin = a, b
        out.append((s + ini / 1000, s + fin / 1000))
    return out


def normalizar_volumen(seg: np.ndarray, P: dict) -> np.ndarray:
    rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
    pico = float(np.max(np.abs(seg))) + 1e-9
    ganancia_db = min(P["volumen_objetivo_dbfs"] - 20 * np.log10(rms),
                      P["pico_max_dbfs"] - 20 * np.log10(pico))
    return seg * (10 ** (ganancia_db / 20))


# ---------------------------------------------------------------- whisper

def es_alucinacion(sg, P: dict) -> bool:
    t = normalizar(sg.text)
    if not t or any(a in t for a in P["alucinaciones"]):
        return True
    if set(tokens(t)) <= {"hola", "thonny"}:  # "hola thonny" repetido es legítimo, no alucinación
        return False
    if sg.compression_ratio > P["compresion_max"]:
        return True
    return sg.no_speech_prob > P["no_speech_max"] and sg.avg_logprob < -1.0


def transcribir(modelo, x, sr, tramos, P, W) -> tuple[list[Palabra], list[str]]:
    """Transcribe cada tramo por separado. Devuelve las palabras (tiempos absolutos) y el texto de cada tramo."""
    dur = len(x) / sr
    palabras, textos = [], []
    for k, (s, e) in enumerate(tramos):
        # margen sin pisar el tramo vecino, para no transcribir dos veces la misma palabra
        lim_a = (tramos[k - 1][1] + s) / 2 if k > 0 else 0.0
        lim_b = (e + tramos[k + 1][0]) / 2 if k + 1 < len(tramos) else dur
        a, b = max(lim_a, s - P["margen_tramo_s"]), min(lim_b, e + P["margen_tramo_s"])
        segs, _ = modelo.transcribe(x[int(a * sr):int(b * sr)], language=W["idioma"],
                                    beam_size=W["beam_size"], word_timestamps=True,
                                    condition_on_previous_text=False)
        texto = []
        for sg in segs:
            if es_alucinacion(sg, P):
                continue
            texto.append(sg.text.strip())
            for w in sg.words or []:
                c = canonica(w.word)
                if c:
                    palabras.append(Palabra(w.word.strip(), c, a + w.start, a + w.end))
        textos.append(" ".join(texto))
    palabras.sort(key=lambda p: p.ini)
    return palabras, textos


# ---------------------------------------------------------------- guion

def unidades_del_guion(g: dict) -> list[Unidad]:
    act = config()["activacion"]["frase"]
    modo = g["modo"]
    if modo == "repetir":
        return [Unidad(g["tipo_frase"], " ".join(g["texto"].split()))] * g["veces"]
    if modo == "ocurrencias":
        return [Unidad(g["tipo_frase"], g["texto"])] * g["esperadas"]
    if modo == "pares":
        return [u for c in g["comandos"] for u in (Unidad("activacion", act), Unidad("comando", c))]
    if modo == "frases":
        out = []
        for f in g["frases"]:
            if normalizar(f).startswith(normalizar(act) + " "):
                out += [Unidad("activacion", act), Unidad("comando", re.split(r",\s*", f, maxsplit=1)[1])]
            else:
                out.append(Unidad(g["tipo_frase"], f))
        return out
    raise ValueError(f"modo desconocido en guion: {modo}")


# ---------------------------------------------------------------- emparejar

def por_silencio(unidades, tramos, textos, P) -> list[Clip] | None:
    """Si hay un tramo por frase y el texto cuadra en promedio, usa los tramos como cortes."""
    if len(tramos) != len(unidades):
        return None
    sims = [similitud(u.texto, t) for u, t in zip(unidades, textos)]
    if np.mean(sims) < P["similitud_min"]:
        return None
    return [Clip(u.tipo_frase, u.texto, s, e) for u, (s, e) in zip(unidades, tramos)]


def por_alineacion(unidades, palabras, P) -> tuple[list[Clip], list[str]]:
    """Alinea (programación dinámica) las palabras esperadas con las reconocidas."""
    esp = [(k, t) for k, u in enumerate(unidades) for t in tokens(u.texto)]
    n, m = len(esp), len(palabras)
    S = np.zeros((n, m))
    for i, (_, a) in enumerate(esp):
        for j, p in enumerate(palabras):
            r = 100.0 if a == p.canon else fuzz.ratio(a, p.canon)
            S[i, j] = r / 100 if r >= P["similitud_palabra"] else 0.0
    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = max(D[i - 1, j], D[i, j - 1],
                          D[i - 1, j - 1] + S[i - 1, j - 1] if S[i - 1, j - 1] > 0 else 0)
    asignadas: dict[int, list[int]] = {}
    i, j = n, m
    while i > 0 and j > 0:
        if S[i - 1, j - 1] > 0 and np.isclose(D[i, j], D[i - 1, j - 1] + S[i - 1, j - 1]):
            asignadas.setdefault(esp[i - 1][0], []).append(j - 1)
            i, j = i - 1, j - 1
        elif np.isclose(D[i, j], D[i - 1, j]):
            i -= 1
        else:
            j -= 1

    clips, faltantes = [], []
    for k, u in enumerate(unidades):
        idx = sorted(set(asignadas.get(k, [])))
        if not idx:
            faltantes.append(f"{k + 1}: {u.texto}")
            continue
        grupos, actual = [], [idx[0]]
        for a, b in zip(idx, idx[1:]):
            if palabras[b].ini - palabras[a].fin > P["separacion_grupo_s"]:
                grupos.append(actual)
                actual = []
            actual.append(b)
        grupos.append(actual)
        g = max(grupos, key=len)
        clips.append(Clip(u.tipo_frase, u.texto, palabras[g[0]].ini, palabras[g[-1]].fin,
                          cobertura=len(g) / len(tokens(u.texto))))
    return clips, faltantes


def por_ocurrencias(g, palabras) -> list[Clip]:
    """Cada "hola" (+ "thonny" si viene enseguida) o "thonny" suelto es una activación."""
    def thonny_cerca(i):
        """Índice del "thonny" que sigue al "hola" de la posición i (admite una palabra de
        relleno en medio, como en "hola a tome"), o None."""
        for j in (i + 1, i + 2):
            if j >= len(palabras) or palabras[j].canon == "hola":
                return None
            if palabras[j].canon == "thonny":
                return j if palabras[j].ini - palabras[i].fin < 1.0 else None
        return None

    clips, i = [], 0
    while i < len(palabras):
        p = palabras[i]
        sig = palabras[i + 1] if i + 1 < len(palabras) else None
        if p.canon == "hola" and (j := thonny_cerca(i)) is not None:
            clips.append(Clip(g["tipo_frase"], g["texto"], p.ini, palabras[j].fin))
            i = j + 1
            continue
        if p.canon in ("hola", "thonny"):
            fin = p.fin + 0.6 if p.canon == "hola" else p.fin
            if sig:
                fin = min(fin, sig.ini)
            ini = p.ini - 0.4 if p.canon == "thonny" else p.ini
            clips.append(Clip(g["tipo_frase"], g["texto"], ini, fin, parcial=True))
        i += 1
    return clips


def habla_libre(clips, palabras, P) -> list[Clip]:
    """Trozos de habla libre después de la última frase esperada."""
    desde = max((c.fin for c in clips), default=0.0) + 0.3
    resto = [p for p in palabras if p.ini >= desde]
    trozos, actual = [], []
    for p in resto:
        if actual and (p.ini - actual[-1].fin > 0.6 or p.fin - actual[0].ini > P["habla_libre_max_s"]):
            trozos.append(actual)
            actual = []
        actual.append(p)
    if actual:
        trozos.append(actual)
    return [Clip("libre", HABLA_LIBRE, t[0].ini, t[-1].fin) for t in trozos if t[-1].fin - t[0].ini >= 1.0]


def ajustar_cortes(clips, x, sr, P, tramos) -> list[Clip]:
    """Lleva cada corte al borde de la voz (tramo por energía) y agrega márgenes,
    sin pisar al clip vecino (se corta en el punto medio)."""
    clips = sorted(clips, key=lambda c: c.ini)
    crudos = [(c.ini, c.fin) for c in clips]
    dur = len(x) / sr
    tol = 0.3

    def tramo_de(t):
        return next(((s, e) for s, e in tramos if s - tol <= t <= e + tol), None)

    for k in range(len(crudos)):
        ini, fin = crudos[k]
        prev_fin = crudos[k - 1][1] if k > 0 else 0.0
        sig_ini = crudos[k + 1][0] if k + 1 < len(crudos) else dur
        if (t := tramo_de(fin)) is not None:
            fin = max(fin, min(t[1], sig_ini))
        if (t := tramo_de(ini)) is not None:
            ini = min(ini, max(t[0], prev_fin))
        crudos[k] = (ini, fin)

    # Punto de corte entre cada par de clips vecinos: el valle de energía cerca del punto medio
    # (las marcas de tiempo de Whisper fallan ~0.2 s cuando las frases van pegadas).
    cortes = [valle_energia(x, sr, crudos[k][1], crudos[k + 1][0], P["busqueda_valle_s"])
              for k in range(len(crudos) - 1)]
    for k, c in enumerate(clips):
        ini, fin = crudos[k]
        lim_ini = cortes[k - 1] if k > 0 else 0.0
        lim_fin = cortes[k] if k + 1 < len(clips) else dur
        c.ini = max(ini - P["margen_inicio_s"], lim_ini, 0.0)
        c.fin = min(fin + P["margen_fin_s"], lim_fin, dur)
    return clips


def valle_energia(x, sr, fin_prev, ini_sig, busqueda) -> float:
    """Instante de menor energía (ventanas de 20 ms) entre dos frases vecinas."""
    a = max(0.0, min(fin_prev, ini_sig) - busqueda)
    b = min(len(x) / sr, max(fin_prev, ini_sig) + busqueda)
    trozo = x[int(a * sr):int(b * sr)]
    paso, ventana = int(0.01 * sr), int(0.02 * sr)
    if len(trozo) < ventana:
        return (fin_prev + ini_sig) / 2
    energia = [np.mean(trozo[i:i + ventana] ** 2) for i in range(0, len(trozo) - ventana + 1, paso)]
    return a + (int(np.argmin(energia)) * paso + ventana / 2) / sr


# ---------------------------------------------------------------- principal

def buscar_audios(cfg, filtro_h, filtro_a):
    crudos = ruta("crudos")
    encontrados, ignorados = [], []
    carpetas = [(h, "miembro", crudos / h) for h in cfg["hablantes"]]
    if (crudos / "ajenos").is_dir():
        carpetas += [(d.name, "ajeno", d) for d in sorted((crudos / "ajenos").iterdir()) if d.is_dir()]
    for h, tipo, d in carpetas:
        if filtro_h and h not in filtro_h:
            continue
        if not d.is_dir():
            print(f"Aviso: no existe {d.relative_to(RAIZ)}")
            continue
        for f in sorted(d.iterdir()):
            mt = PATRON_AUDIO.match(f.name)
            if not mt:
                ignorados.append(str(f.relative_to(RAIZ)))
                continue
            n = int(mt.group(1))
            if filtro_a and n not in filtro_a:
                continue
            encontrados.append((h, tipo, n, f))
    return encontrados, ignorados


def procesar(modelo, h, tipo, n, archivo, guion, cfg):
    P, W = cfg["preparacion"], cfg["whisper"]
    sr = P["sample_rate"]
    g = guion[tipo].get(f"audio{n}")
    if g is None:
        return None, {"error": f"audio{n} no está en el guion para tipo {tipo}"}
    x = cargar(archivo, sr)
    dur = len(x) / sr
    tramos = tramos_por_silencio(x, sr, P)
    unidades = unidades_del_guion(g)
    faltantes: list[str] = []

    if g["modo"] == "ocurrencias":
        tramos = subdividir(x, sr, tramos, P)
        palabras, textos = transcribir(modelo, x, sr, tramos, P, W)
        clips, metodo = por_silencio(unidades, tramos, textos, P), "silencio"
        if clips is None:
            clips, metodo = por_ocurrencias(g, palabras), "ocurrencias"
    else:
        palabras, textos = transcribir(modelo, x, sr, tramos, P, W)
        clips, metodo = por_silencio(unidades, tramos, textos, P), "silencio"
        if clips is None:
            clips, faltantes = por_alineacion(unidades, palabras, P)
            metodo = "alineacion"
        if g.get("habla_libre"):
            clips += habla_libre(clips, palabras, P)

    clips = ajustar_cortes(clips, x, sr, P, tramos)
    filas, descartados = [], 0
    for c in clips:
        if c.fin - c.ini < P["duracion_min_s"]:
            descartados += 1
            continue
        oido = " ".join(p.texto for p in palabras if c.ini <= (p.ini + p.fin) / 2 <= c.fin)
        sim = 100.0 if c.frase_esperada == HABLA_LIBRE else similitud(c.frase_esperada, oido)
        revisar = sim < P["similitud_min"] or c.cobertura < P["cobertura_min"] or c.parcial
        filas.append({"clip": c, "transcripcion": oido, "similitud": round(sim), "revisar": int(revisar)})

    resumen = {
        "archivo": str(archivo.relative_to(RAIZ)), "duracion_s": round(dur, 1), "metodo": metodo,
        "tramos": len(tramos),
        "esperadas": g.get("esperadas", len(unidades)) if g["modo"] == "ocurrencias" else len(unidades),
        "clips": len(filas), "revisar": sum(f["revisar"] for f in filas),
        "faltantes": faltantes, "descartados_cortos": descartados,
    }
    return (x, sr, metodo, filas), resumen


def leer_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hablante", action="append", help="procesar solo este hablante (se puede repetir)")
    ap.add_argument("--audio", type=int, action="append", help="procesar solo este número de audio")
    ap.add_argument("--sin-escribir", action="store_true", help="solo mostrar el reporte, sin guardar clips")
    ap.add_argument("--detalle", action="store_true", help="mostrar cada clip con su corte y transcripción")
    args = ap.parse_args()

    cfg = config()
    with open(RAIZ / cfg["rutas"]["guion"], encoding="utf-8") as f:
        guion = yaml.safe_load(f)
    audios, ignorados = buscar_audios(cfg, args.hablante, args.audio)
    if ignorados:
        print("Archivos ignorados (el nombre no es audio<N>.<ext>):\n  " + "\n  ".join(ignorados))
    if not audios:
        sys.exit("No hay audios para procesar.")

    from faster_whisper import WhisperModel
    W = cfg["whisper"]
    modelo = WhisperModel(W["modelo"], device=W["dispositivo"], compute_type=W["compute_type"])

    csv_path = ruta("clips_csv")
    previas = leer_csv(csv_path)
    previas_por_ruta = {r["ruta_celular"]: r for r in previas}
    procesados, nuevas, resumenes, sobrantes = set(), [], [], []

    print(f"\n{'audio':28s} {'método':12s} {'esperadas':>9s} {'clips':>6s} {'revisar':>8s} {'faltan':>7s}")
    for h, tipo, n, archivo in audios:
        res, resumen = procesar(modelo, h, tipo, n, archivo, guion, cfg)
        resumen.update({"hablante": h, "tipo": tipo, "audio": n})
        resumenes.append(resumen)
        if res is None:
            print(f"{h}/audio{n}: {resumen['error']}")
            continue
        print(f"{h + '/' + archivo.name:28s} {resumen['metodo']:12s} {resumen['esperadas']:9d} "
              f"{resumen['clips']:6d} {resumen['revisar']:8d} {len(resumen['faltantes']):7d}")
        for ftxt in resumen["faltantes"]:
            print(f"    no encontrada -> {ftxt}")
        x, sr, metodo, filas = res
        if args.detalle:
            for k, fila in enumerate(filas, 1):
                c = fila["clip"]
                marca = "REVISAR" if fila["revisar"] else ""
                print(f"    {k:02d} {c.ini:6.2f}-{c.fin:6.2f}s {c.tipo_frase:10s} sim {fila['similitud']:3d}  "
                      f"{c.frase_esperada[:40]:40s} | {fila['transcripcion'][:50]} {marca}")
        procesados.add((h, str(n)))
        carpeta = ruta("clips_celular") / h
        nombres = set()
        for k, fila in enumerate(filas, 1):
            c = fila["clip"]
            nombre = f"{h}_a{n}_{k:02d}.wav"
            nombres.add(nombre)
            destino = carpeta / nombre
            rel = destino.relative_to(RAIZ).as_posix()
            vieja = previas_por_ruta.get(rel, {})
            mismo_corte = vieja.get("inicio_s") == f"{c.ini:.2f}" and vieja.get("fin_s") == f"{c.fin:.2f}"
            seg = normalizar_volumen(x[int(c.ini * sr):int(c.fin * sr)], cfg["preparacion"])
            # Control final: el clip ya cortado, transcrito solo, debe contener su frase
            oido_clip, sim_clip = "", 100
            if c.frase_esperada != HABLA_LIBRE:
                segs, _ = modelo.transcribe(seg, language=W["idioma"], beam_size=W["beam_size"],
                                            condition_on_previous_text=False)
                oido_clip = " ".join(s.text.strip() for s in segs)
                sim_clip = round(similitud(c.frase_esperada, oido_clip))
                if sim_clip < cfg["preparacion"]["similitud_clip_min"]:
                    fila["revisar"] = 1
            if args.detalle and fila["revisar"]:
                print(f"    revisar {nombre}: esperado '{c.frase_esperada[:40]}' | clip solo: '{oido_clip[:50]}' ({sim_clip})")
            if not args.sin_escribir:
                carpeta.mkdir(parents=True, exist_ok=True)
                sf.write(destino, seg, sr, subtype="PCM_16")
            nuevas.append({
                "hablante": h, "tipo": tipo, "audio": n, "n": k, "tipo_frase": c.tipo_frase,
                "frase_esperada": c.frase_esperada, "duracion_s": f"{c.fin - c.ini:.2f}",
                "ruta_celular": rel,
                "ruta_microfono": vieja.get("ruta_microfono", "") if mismo_corte else "",
                "particion": "test" if (tipo == "miembro" and n == 5) else vieja.get("particion", ""),
                "transcripcion": fila["transcripcion"], "similitud": fila["similitud"],
                "transcripcion_clip": oido_clip, "similitud_clip": sim_clip,
                "revisar": fila["revisar"], "metodo": metodo,
                "inicio_s": f"{c.ini:.2f}", "fin_s": f"{c.fin:.2f}",
            })
        total_revisar = sum(f["revisar"] for f in filas)
        if total_revisar != resumen["revisar"]:
            print(f"    -> tras el control final quedan {total_revisar} clips para revisar")
            resumen["revisar"] = total_revisar
        if carpeta.is_dir():
            sobrantes += [str(p.relative_to(RAIZ)) for p in carpeta.glob(f"{h}_a{n}_*.wav") if p.name not in nombres]

    if args.sin_escribir:
        print("\n--sin-escribir: no se guardó nada.")
        return

    filas_csv = [r for r in previas if (r["hablante"], r["audio"]) not in procesados] + nuevas
    filas_csv.sort(key=lambda r: (r["tipo"], r["hablante"], int(r["audio"]), int(r["n"])))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas_csv)

    salida = ruta("resultados") / "preparar_audios.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"),
                                  "audios": resumenes, "ignorados": ignorados},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(nuevas)} clips -> {csv_path.relative_to(RAIZ)}  |  resumen -> {salida.relative_to(RAIZ)}")
    if sobrantes:
        print("Ojo: estos clips son de una corrida anterior y ya no corresponden (no los borré):\n  "
              + "\n  ".join(sobrantes))


if __name__ == "__main__":
    main()
