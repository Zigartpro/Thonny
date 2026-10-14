"""Fase 4: embeddings ECAPA y clasificadores de hablante.

  Encoder         ECAPA-TDNN de SpeechBrain; embeddings normalizados (192 dims), con caché en disco.
  Coseno          promedio (centroide) de cada persona + umbral sobre la similitud coseno.
  RegLog          regresión logística con las personas + una clase "ajeno" + umbral de confianza.
Ambos devuelven, para cada clip, (persona más probable, puntaje). Thonny acepta si puntaje >= umbral.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, config  # noqa: E402

AJENO = "ajeno"


class Encoder:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        H = config()["hablante"]
        self.modelo = EncoderClassifier.from_hparams(
            source=H["modelo"], savedir=str(RAIZ / H["carpeta"]),
            run_opts={"device": H["dispositivo"]}, local_strategy=LocalStrategy.COPY)
        self.cache_path = RAIZ / "cache" / "embeddings.npz"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            d = np.load(self.cache_path)
            self.cache = {k: d[k] for k in d.files}
        self._nuevos = 0

    def de_audio(self, wav: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = self.modelo.encode_batch(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0))
        e = e.squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    def de_archivo(self, ruta: str, variante: str = "", transformar=None) -> np.ndarray:
        """Embedding de un WAV de 16 kHz; `transformar(wav, semilla)` aplica un aumento (con caché propia)."""
        p = Path(ruta)
        clave = hashlib.md5(f"{p.resolve()}|{p.stat().st_mtime_ns}|{variante}".encode()).hexdigest()
        if clave not in self.cache:
            wav, sr = sf.read(p, dtype="float32")
            assert sr == 16000, f"{p} no está a 16 kHz"
            if transformar is not None:
                wav = transformar(wav, int(clave[:8], 16))
            self.cache[clave] = self.de_audio(wav)
            self._nuevos += 1
            if self._nuevos % 200 == 0:
                self.guardar()
        return self.cache[clave]

    def matriz(self, filas, variante: str = "", transformar=None) -> np.ndarray:
        return np.stack([self.de_archivo(f["ruta"], variante, transformar) for f in filas])

    def guardar(self) -> None:
        if self._nuevos:
            self.cache_path.parent.mkdir(exist_ok=True)
            np.savez(self.cache_path, **self.cache)
            self._nuevos = 0


class Coseno:
    nombre = "coseno"

    def fit(self, X: np.ndarray, y: list[str]):
        self.personas = sorted({e for e in y if e != AJENO})
        C = np.stack([X[[i for i, e in enumerate(y) if e == p]].mean(axis=0) for p in self.personas])
        self.centroides = C / np.linalg.norm(C, axis=1, keepdims=True)
        return self

    def puntuar(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        S = X @ self.centroides.T
        k = S.argmax(axis=1)
        return [self.personas[i] for i in k], S.max(axis=1)


class RegLog:
    nombre = "reglog"

    def fit(self, X: np.ndarray, y: list[str]):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=config()["verificacion"]["C_reglog"], max_iter=5000,
                                      class_weight="balanced").fit(X, y)
        self.personas = [c for c in self.clf.classes_ if c != AJENO]
        self.idx = [list(self.clf.classes_).index(p) for p in self.personas]
        return self

    def puntuar(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        P = self.clf.predict_proba(X)[:, self.idx]  # probabilidad de cada persona
        k = P.argmax(axis=1)
        return [self.personas[i] for i in k], P.max(axis=1)


CLASIFICADORES = {"coseno": Coseno, "reglog": RegLog}
