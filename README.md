# Thonny

Asistente de voz con palabra de activación **"Hola Thonny"** (proyecto de la materia PLN).
Reconoce la voz de 3 personas, responde con su voz clonada y prende o apaga 4 aparatos
de 110 V (bombillo, ventilador, cautín y plancha de cabello) mediante una ESP32-S3 y un
módulo de relés. Todo corre local.

El plan completo está en [PLAN_THONNY.md](PLAN_THONNY.md).

## Estructura

```
firmware/     código de la ESP32-S3 (Arduino y MicroPython)
  pruebas_hardware/  pruebas sueltas de micrófono, parlante y eco
pc/           servidor de Thonny y scripts
datos/        audios crudos, clips, referencias TTS, CSVs
  slr72_sinteticos/  impostores generados con F5-TTS (copia de datasets/slr72/sinteticos)
voces/        respuestas pregeneradas por hablante
modelos/      ECAPA ajustado y clasificador de intenciones
cache/        embeddings precalculados
resultados/   métricas y figuras generadas por los scripts
config.yaml   configuración (hablantes, rutas, puertos, umbrales…)
```

El repo es **privado** porque contiene las grabaciones de voz de los hablantes.

## Restaurar en un PC nuevo

Lo que no está en el repo se vuelve a descargar o regenerar:

1. **Entornos virtuales:** ver la sección *Entorno* más abajo.
2. **Modelos de HuggingFace** (faster-whisper-medium, F5-Spanish, vocos): se descargan solos la
   primera vez que corren los scripts.
3. **Dataset SLR72** (OpenSLR, español colombiano, CC BY-SA 4.0): descargar `es_co_female.zip` y
   `es_co_male.zip` de <https://openslr.org/72/> en `C:/Users/<usuario>/datasets/slr72/` y
   ejecutar `python pc\preparar_ajenos.py`. Luego copiar `datos/slr72_sinteticos/*` a
   `datasets/slr72/sinteticos/`. Si cambia la ruta, ajustar `rutas.dataset_ajenos` en
   `config.yaml` y las rutas de `datos/ajenos*.csv`.
4. **Credenciales WiFi:** copiar `firmware/thonny_esp32/secrets.h.ejemplo` como `secrets.h`
   y `firmware/micropython/secrets.py.ejemplo` como `secrets.py`; no se suben al repo.

## Entorno (Windows 11)

El entorno virtual vive **fuera de OneDrive** para que no se sincronicen gigas de librerías:

```powershell
py install 3.11
py -V:3.11 -m venv C:\Users\duvan\venvs\thonny
C:\Users\duvan\venvs\thonny\Scripts\Activate.ps1
pip install -r requirements.txt
```

F5-TTS (voz clonada, Fase 5) va en un entorno aparte, porque trae muchas dependencias y nunca
corre al mismo tiempo que Thonny:

```powershell
py -V:3.11 -m venv C:\Users\duvan\venvs\thonny-tts
C:\Users\duvan\venvs\thonny-tts\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
C:\Users\duvan\venvs\thonny-tts\Scripts\python.exe -m pip install f5-tts pyyaml
```

ffmpeg debe estar en el PATH (`winget install Gyan.FFmpeg`).

faster-whisper necesita cuBLAS (CUDA 12) y cuDNN 9. No se instalan aparte: se usan las DLL
que trae PyTorch (`torch/lib`), que `pc/cuda_dlls.py` agrega al buscador de DLL.

Verificar el entorno:

```powershell
python pc\verificar_entorno.py
```
