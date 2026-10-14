"""Hace visibles para faster-whisper (CTranslate2) las DLL de CUDA 12 y cuDNN 9 que trae PyTorch.

Importar este módulo antes de `faster_whisper`. En Windows CTranslate2 busca
cublas64_12.dll y cudnn*64_9.dll en el PATH; PyTorch ya las trae en torch/lib.
"""

import os
import sys
from pathlib import Path


def registrar() -> None:
    if sys.platform != "win32":
        return
    import torch  # noqa: F401  (carga sus propias DLL de CUDA)

    lib = Path(torch.__file__).parent / "lib"
    if lib.is_dir():
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")


registrar()
