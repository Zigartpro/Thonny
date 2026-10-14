# PLAN_THONNY.md — Thonny desde cero

## Qué es Thonny

Thonny es un asistente de voz con palabra de activación **"Hola Thonny"** (proyecto de la materia PLN). Se construye desde cero en este PC; del proyecto anterior (Doby) solo se toma el diseño general, no hay código previo.

Thonny:

- se activa solo con la voz de 3 personas (nombres en la configuración) y no con voces ajenas;
- responde hablando con la voz clonada de quien le habló (F5-TTS, respuestas pregeneradas);
- prende y apaga 4 aparatos de 110 V con un módulo de relés de 4 canales: bombillo, ventilador, cautín y plancha de cabello.

Si quien habla no es uno de los 3, Thonny no se activa (solo lo registra en el log). La voz por defecto se usa para mensajes del sistema ("Thonny listo") y como respaldo si falta un audio clonado.

## Equipo y restricciones

- PC: AMD Ryzen 5 7235HS (4 núcleos, 8 hilos), ~11.7 GB de RAM, NVIDIA RTX 3050 con **6 GB de VRAM**, Windows 11.
- Todo corre local; nada de APIs en la nube mientras Thonny funciona.
- **Presupuesto de VRAM:** en tiempo real solo se cargan Whisper y la verificación de hablante. F5-TTS se usa aparte, por lotes, para pregenerar las respuestas, y nunca al mismo tiempo que Thonny.
- **RAM:** ya está cerca del límite; cerrar navegador y programas pesados al entrenar o generar voces.

Modelos:

| Tarea | Modelo | Notas |
| --- | --- | --- |
| Voz a texto | faster-whisper `medium`, `compute_type="int8_float16"`, idioma `es` | Si va lento o no cabe, bajar a `small`. En Windows necesita cuBLAS (CUDA 12) y cuDNN 9; seguir las instrucciones de faster-whisper para Windows. |
| Detección de voz | Silero VAD | Corta el audio continuo en frases. |
| Hablante | SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) + clasificador propio sobre los embeddings | Corre bien en CPU. |
| Intenciones | TF-IDF + LogisticRegression (scikit-learn), entrenado con dataset propio | |
| Voz clonada | F5-TTS con el checkpoint en español `jpgallegoar/F5-Spanish` (licencia cc0-1.0) | Solo para pregenerar respuestas. |
| Voz por defecto | gTTS pregenerado (necesita internet una sola vez al generar) | Alternativa 100 % local: Piper TTS en español. |

## Hardware

- ESP32-S3-N16R8 alimentada con un regulador a 5 V.
- Micrófono INMP441 (I2S, entrada) y amplificador MAX98357 (I2S, salida) con parlante de 8 Ω y 1.5 W. La ESP32-S3 tiene dos controladores I2S: uno para cada uno.
- Módulo de relés de 4 canales a 5 V (probablemente activo en bajo: verificarlo).
- Aparatos de 110 V: bombillo, ventilador, cautín y plancha de cabello; los dos últimos por tomacorrientes hembra alimentados a través de los relés.

Reglas de pines (ESP32-S3-N16R8):

- No usar GPIO 26–32 (memoria flash) ni 33–37 (PSRAM octal).
- No usar GPIO 19–20 (USB) ni los de arranque 0, 3, 45 y 46.
- Al arrancar, los pines de los relés deben quedar en el nivel de "apagado" antes de configurarlos como salida, para que ningún aparato se prenda solo al encender la placa.

## Arquitectura

```
INMP441 → ESP32-S3 ──(WiFi, TCP) audio PCM 16 kHz──► PC
                                                     1. Silero VAD corta la frase
                                                     2. faster-whisper la pasa a texto
                                                     3. ¿Dice "Hola Thonny"? (variantes + rapidfuzz)
                                                     4. ECAPA: ¿es uno de los 3? ¿cuál?
                                                     5. Clasificador de intenciones
                                                     6. Respuesta pregenerada en la voz de quien habló
PC ──(WiFi, TCP) orden de relé + audio de respuesta──► ESP32-S3 → relés y MAX98357
```

Propuesta de protocolo (la ESP32 es servidor TCP; documentarlo en `PROTOCOLO.md`):

- Puerto 5000, ESP32 → PC: audio del micrófono continuo, PCM 16 bits little-endian, 16 kHz, mono.
- Puerto 5001, PC → ESP32: líneas JSON de control, por ejemplo `{"cmd":"rele","n":1,"on":true}` y `{"cmd":"play","bytes":N}` seguido de N bytes de PCM 16 kHz para el parlante. La ESP32 responde `{"ok":true}` o un error.
- Mientras Thonny habla, el PC descarta el audio del micrófono para no escucharse a sí mismo.
- Credenciales de WiFi en `firmware/secrets.h`, fuera de git.

Flujo de conversación:

1. Thonny escucha siempre; el VAD corta cada frase.
2. Si la frase trae "Hola Thonny", verifica al hablante.
3. Si es uno de los 3:
    - si la frase ya trae el comando ("Hola Thonny, prende la luz"), lo ejecuta y confirma;
    - si no, responde "¿Qué necesitas?" en la voz de esa persona y escucha el comando durante unos 6 s, verificando que sea la misma persona.
4. Si no es uno de los 3, no hace nada y lo registra en el log.

Respuestas pregeneradas por voz (editables en `respuestas.yaml`): "¿Qué necesitas?", una confirmación por cada comando ("Listo, prendí la luz", "Listo, apagué el ventilador"…) y "No te entendí, ¿me repites?".

## Los datos

Las 3 personas no están presentes. Grabaron con la grabadora del celular y mandaron los audios por WhatsApp (pueden llegar como .m4a, .opus, .ogg, .mp3, .aac o .wav). Cada audio es una grabación larga con varias frases separadas por pausas de unos 2 s.

| Origen | Audio | Contenido | Uso |
| --- | --- | --- | --- |
| Cada una de las 3 personas | 1 | Texto de referencia leído 3 veces | Referencia F5-TTS (mejor toma) y entrenamiento de hablante |
| | 2 | "Hola Thonny" ×15 con variaciones de volumen, velocidad y distancia | Entrenamiento de hablante |
| | 3 | 20 frases de comandos (lista abajo) | Entrenamiento de hablante |
| | 4 | 8 frases sueltas + ~30 s de habla libre | Entrenamiento de hablante |
| | 5 | 8 × ("Hola Thonny", pausa de 3 s, comando) | **Solo prueba y demo. Nunca entra al entrenamiento.** |
| Ajenos (3–4 personas) | 1 | "Hola Thonny" ×15 | Rechazo (entrenamiento o prueba según partición) |
| | 2 | 4 × ("Hola Thonny", pausa de 3 s, comando) | Rechazo (entrenamiento o prueba según partición) |
| En vivo | — | Jenpol y, si se puede, 1–2 personas más diciendo "Hola Thonny" y comandos directo al INMP441 | Rechazo de voces en persona |
| Opcional | — | Voces de un dataset público en español (por ejemplo Common Voice) | Más voces ajenas para entrenar y calibrar |

Texto del Audio 1:

> Hola, soy Thonny, el asistente de voz de este proyecto. Puedo prender y apagar el bombillo, el ventilador, el cautín y la plancha cuando me lo pidas.

Frases del Audio 3, en orden:

1. Prende el bombillo
2. Enciende la luz
3. Apaga el bombillo
4. Apaga la luz
5. Prende el ventilador
6. Enciende el ventilador
7. Apaga el ventilador
8. Ya apaga el ventilador
9. Prende el cautín
10. Enciende el cautín
11. Apaga el cautín
12. Desconecta el cautín
13. Prende la plancha
14. Enciende la plancha de cabello
15. Apaga la plancha
16. Desconecta la plancha de cabello
17. Hola Thonny, prende la luz
18. Hola Thonny, apaga el ventilador
19. Hola Thonny, prende el cautín
20. Hola Thonny, apaga la plancha

Comandos del Audio 5, en orden (cada uno precedido de "Hola Thonny" y una pausa de 3 s): prende la luz, apaga la luz, prende el ventilador, apaga el ventilador, prende el cautín, apaga el cautín, prende la plancha, apaga la plancha.

Audio 2 de los ajenos: prende la luz, apaga el ventilador, prende el cautín, apaga la plancha (cada uno precedido de "Hola Thonny" y una pausa de 3 s).

### Dos versiones de cada clip

- `celular`: el archivo original, convertido.
- `microfono`: el mismo clip reproducido por el parlante del PC y captado por el INMP441. Así suena lo que Thonny escucha en la presentación, donde se van a reproducir audios.

Las voces ajenas grabadas con celular pasan por el mismo camino que las de las 3 personas, para que el modelo no aprenda a distinguir "audio por parlante" de "voz en vivo" en lugar de las voces.

### Carpetas

```
firmware/                 código de la ESP32 (Arduino)
pc/                       servidor de Thonny y scripts
datos/
  crudos/<hablante>/audio1.* ... audio5.*
  crudos/ajenos/<nombre>/audio1.*, audio2.*
  crudos/vivo/<nombre>/*.wav
  clips/celular/<hablante>/<hablante>_a<N>_<NN>.wav
  clips/microfono/<hablante>/<hablante>_a<N>_<NN>.wav
  referencia_tts/<hablante>.wav + <hablante>.txt
  intenciones.csv
  clips.csv
voces/<hablante>/*.wav    respuestas pregeneradas
resultados/
config.yaml
respuestas.yaml
```

`clips.csv` lleva: hablante, tipo (miembro | ajeno | vivo), audio, n, tipo_frase (activacion | comando | libre), frase_esperada, duracion_s, ruta_celular, ruta_microfono, particion (train | val | test).

## Reglas de trabajo

- Una fase a la vez. Al terminar cada una: resumen de cambios, cómo probarla, y esperar mi aprobación.
- Nombres de hablantes, rutas, puertos, IP, umbrales y variantes de la palabra de activación van en `config.yaml`, no quemados en el código.
- Ninguna métrica se escribe a mano: todo número sale de correr un script y queda guardado en `resultados/`.
- El Audio 5 de las 3 personas y todo lo marcado como `test` nunca entra al entrenamiento ni a la calibración del umbral. Ponlo como una aserción en el código.
- Antes de borrar archivos, pregúntame.
- Antes de cargar firmware a la ESP32, muéstrame el mapeo de pines y espera mi confirmación.
- Nada de 110 V hasta que los relés funcionen bien sin carga.

## Fase 0 — Entorno

1. Crear la estructura de carpetas, `git init`, `.gitignore` (modelos, datos de voz, `secrets.h`, entorno virtual), `README.md` y un `CLAUDE.md` con el resumen de este plan y las reglas de trabajo.
2. Entorno virtual con Python 3.11; PyTorch con CUDA; faster-whisper, silero-vad, speechbrain, scikit-learn, rapidfuzz, pydub, sounddevice, pyyaml; ffmpeg en el PATH.
3. `pc/verificar_entorno.py`: muestra la GPU, la VRAM libre y las versiones, y transcribe un WAV de prueba con faster-whisper `medium` en `int8_float16`, midiendo tiempo y VRAM usada.

## Fase 1 — Firmware de la ESP32

Arduino (núcleo ESP32 3.x), compilado y cargado con `arduino-cli` para poder hacerlo desde la terminal.

1. Proponer el mapeo de pines (INMP441, MAX98357, IN1–IN4) en `firmware/config.h` siguiendo las reglas de pines, y esperar mi confirmación según mi montaje.
2. WiFi y servidor TCP con el protocolo de la sección Arquitectura.
3. Relés en estado apagado desde el arranque; indicar si el módulo es activo en bajo.
4. Pruebas por partes, cada una con un script del PC:
    - `pc/prueba_mic.py`: graba 5 s del INMP441 y guarda un WAV.
    - `pc/prueba_reles.py`: prende y apaga cada relé, **sin carga de 110 V**.
    - `pc/prueba_parlante.py`: envía un WAV y suena por el MAX98357.
5. Escribir `PROTOCOLO.md`.

## Fase 2 — Servidor del PC (sin verificación de hablante todavía)

1. Recibir el audio, cortar frases con Silero VAD y transcribir con faster-whisper.
2. Activación: lista de variantes en `config.yaml` (thonny, tony, toni, tonny, thony, tonni…) con coincidencia aproximada (rapidfuzz) y `initial_prompt="Hola Thonny"` en Whisper.
3. Clasificador de intenciones:
    - Dataset propio en `datos/intenciones.csv` con 20–30 frases por intención: prender y apagar cada aparato (8 intenciones), saludo y desconocido.
    - Sinónimos: luz, bombillo, foco; prende, enciende, activa, pon; apaga, desconecta, quita; plancha, plancha de cabello, plancha del pelo.
    - Normalizar el texto (minúsculas, sin tildes) e incluir errores típicos de Whisper que salgan en los logs.
    - TF-IDF + LogisticRegression, con exactitud por validación cruzada guardada en `resultados/`.
4. Respuestas temporales con la voz por defecto (gTTS pregenerado).
5. Modo depuración (bandera en `config.yaml`): sin verificación de hablante; imprime transcripción, intención y confianza.
6. Prueba: yo digo "Hola Thonny, prende la luz" y se activa el relé.

## Fase 3 — Herramientas de datos

Se puede hacer antes de que lleguen los audios: pruébalo con audios míos grabados con el celular.

1. `pc/preparar_audios.py`
    - Convierte `datos/crudos/` a WAV 16 kHz mono (ffmpeg) y normaliza el volumen.
    - Corta por silencios (pydub `split_on_silence`, parámetros en `config.yaml`) y descarta clips de menos de 0.4 s.
    - En los audios con "Hola Thonny … comando", la activación y el comando quedan como clips separados y consecutivos (tipo_frase = activacion / comando).
    - Asigna la frase esperada en orden. Si el número de clips no coincide, transcribe con Whisper para emparejar; si aún no cuadra, marca ese audio para revisión manual.
    - Nombra los clips y llena `clips.csv`.
2. `pc/grabar_por_microfono.py`
    - Reproduce cada clip de `clips/celular` por el parlante del PC y graba al mismo tiempo lo que capta el INMP441, por el mismo camino de audio que usa Thonny.
    - Recorta el silencio sobrante y guarda en `clips/microfono` con el mismo nombre.
    - Volumen y pausa entre clips configurables; se puede reanudar si se corta.
3. `pc/grabar_en_vivo.py`: graba frases dichas directo al INMP441 y las guarda en `datos/crudos/vivo/<nombre>/`.
4. `pc/elegir_referencia_tts.py`
    - Corta el Audio 1 de cada persona en sus 3 tomas (versión `celular`), las transcribe con Whisper y me muestra la duración y las diferencias con el texto esperado.
    - Yo escojo la mejor. Se guarda en `referencia_tts/` con su transcripción exacta: si la persona cambió una palabra, la transcripción lo refleja.
5. Partición fija en `clips.csv`:
    - `test`: todo el Audio 5 de las 3 personas; todos los audios de al menos 2 ajenos completos (personas que el modelo nunca ve); al menos una persona en vivo que no esté en `train`.
    - `val`: una parte de los clips de entrenamiento, para escoger el umbral.
    - `train`: Audios 1 a 4 de las 3 personas, el resto de ajenos, las grabaciones en vivo restantes y, si se usa, el dataset público.

## Fase 4 — Reconocimiento de hablante y evaluación

1. Extraer embeddings ECAPA de cada clip (en caché, para no recalcular).
2. Clasificador sobre los embeddings, comparando en `val`:
    - similitud coseno con el promedio de cada persona + umbral de rechazo;
    - regresión logística o SVM con una clase "ajeno" + umbral de confianza.
3. `pc/evaluar.py`, sobre `test`, reportando aparte la versión `microfono` (la de la presentación) y la versión `celular`:
    - aceptación correcta por persona (activaciones de los 3 que se aceptan);
    - identificación correcta (si acierta quién habla);
    - falsa aceptación de ajenos, grabados y en vivo por separado;
    - matriz de confusión;
    - barrido de umbral con el punto de error igual (EER) y el umbral escogido con `val`.

    Guarda `resultados/<experimento>.json`, una figura PNG por experimento y actualiza `resultados/comparacion.md`.
4. Experimentos, todos con el mismo `test` y el umbral escogido con `val`:
    - **E1 — línea base:** un solo clip de "Hola Thonny" por persona, versión `celular`.
    - **E2:** todos los clips de `train`, versión `celular`.
    - **E3:** todos los clips de `train`, versiones `celular` + `microfono`.
    - **E4 (opcional):** E3 con aumentos de datos (ruido de fondo, cambio de volumen).

    Thonny usa el que mejor salga en los resultados. La tabla de `comparacion.md` es la justificación en el informe de por qué hicieron falta todos los audios.
5. Integrar la verificación en el servidor de la Fase 2.

## Fase 5 — Voces de respuesta

Con Thonny cerrado, como proceso aparte.

1. `pc/generar_voces.py` con F5-TTS y el checkpoint `jpgallegoar/F5-Spanish`, usando la referencia de cada persona (Fase 3.4). Medir la VRAM usada; si no cabe en 6 GB, reducir pasos de inferencia o generar por partes.
2. Generar primero una muestra corta por persona y pedirme aprobación antes de generar todo.
3. Generar todas las respuestas de `respuestas.yaml` para las 3 voces en `voces/<hablante>/`.
4. Generar la voz por defecto (gTTS) con el nombre Thonny.

## Fase 6 — Integración y demo

1. Juntar todo en `pc/thonny.py` con la configuración final.
2. `pc/demo.py`: reproduce por el parlante el Audio 5 de cada persona, en orden: clip de activación → espera la respuesta de Thonny → clip de comando → espera a que se accione el relé. Después reproduce clips de ajenos de `test` para mostrar que no se activa.
3. Prueba final: cada persona × cada comando, ajenos grabados, voz en vivo y con ruido de fondo. Resultados en `resultados/prueba_final.md`.
