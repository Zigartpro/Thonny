# Protocolo PC ↔ ESP32-S3

La ESP32 es el servidor. Se le habla por **WiFi** (lo que usa Thonny) o por **cable USB**
(pruebas y grabación de clips). Implementación: [firmware/thonny_esp32/thonny_esp32.ino](firmware/thonny_esp32/thonny_esp32.ino);
cliente del PC: [pc/esp32.py](pc/esp32.py).

## Formato del audio

En los dos sentidos: **PCM de 16 bits con signo, little-endian, 16 kHz, mono**, sin encabezado.

## WiFi

La placa se conecta a la red de `firmware/thonny_esp32/secrets.h` (2.4 GHz) y responde como
`thonny.local` o por la IP que muestra el monitor serie al arrancar.

| Puerto | Sentido | Contenido |
|---|---|---|
| 5000 | ESP32 → PC | Audio del INMP441, continuo, mientras haya un cliente conectado |
| 5001 | PC → ESP32 | Órdenes JSON (una por línea, terminadas en `\n`) y audio para el parlante |

Un solo cliente por puerto; si se conecta otro, reemplaza al anterior.

## Cable USB

Puerto "UART" de la placa (chip CH343), **921600 baudios**, 8N1. Para no reiniciar la placa
al abrir el puerto, el PC deja DTR y RTS en bajo. Acepta las mismas órdenes JSON que el
puerto 5001, más:

| Orden | Efecto |
|---|---|
| `{"cmd":"audio_serial"}` | Responde `{"ok":true,"modo":"audio_serial","sample_rate":16000}` y desde ese momento el cable solo lleva el audio del micrófono (igual que el puerto 5000). Se sale reiniciando la placa (pulso en RTS). |

Mientras el cable no está en modo audio, la placa escribe mensajes de estado legibles (líneas
que no empiezan por `{`); el PC los ignora.

## Órdenes JSON

| Orden | Respuesta | Efecto |
|---|---|---|
| `{"cmd":"ping"}` | `{"ok":true}` | |
| `{"cmd":"rele","n":1,"on":true}` | `{"ok":true}` | Prende (`true`) o apaga (`false`) el relé `n` (1–4) |
| `{"cmd":"todos_off"}` | `{"ok":true}` | Apaga los 4 relés |
| `{"cmd":"play","bytes":N}` + N bytes | `{"ok":true}` al terminar | Reproduce N bytes de PCM por el MAX98357 (N par, máximo 15 s) |
| `{"cmd":"estado"}` | `{"ok":true,"reles":[0,0,0,0],"wifi":true,"ip":"…","rssi":-55,"heap":…}` | |

Errores: `{"ok":false,"error":"<motivo>"}` (orden desconocida, `n` fuera de rango, audio
incompleto si el PC deja de mandar bytes por más de 3 s, etc.).

| Relé | GPIO | Aparato |
|---|---|---|
| 1 | 10 | bombillo |
| 2 | 11 | ventilador |
| 3 | 12 | cautín |
| 4 | 13 | plancha de cabello |

El módulo es activo en bajo (`RELE_ACTIVO_BAJO` en `config.h`). Al arrancar, los pines se
ponen en el nivel de "apagado" antes de configurarse como salida.

## Reglas del PC

- Mientras Thonny reproduce una respuesta, descarta el audio del micrófono para no escucharse a sí mismo.
- Las credenciales de WiFi van en `firmware/thonny_esp32/secrets.h`, fuera de git.
