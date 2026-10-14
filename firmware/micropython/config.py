# Configuración del firmware MicroPython de Thonny (ESP32-S3).
# Mismos pines que firmware/thonny_esp32/config.h.

# Micrófono INMP441 (VDD -> 3V3, GND -> GND, L/R -> GND)
MIC_SCK = 15
MIC_WS = 16
MIC_SD = 17
# El INMP441 entrega 24 bits dentro de palabras de 32. Se desplaza a 16 bits;
# un desplazamiento menor = más ganancia (14 => +12 dB respecto a 16).
MIC_SHIFT = 14

# Amplificador MAX98357 (VIN -> 5V, GND -> GND, SD y GAIN sin conectar)
AMP_BCLK = 5
AMP_LRC = 6
AMP_DIN = 7

# Relés: IN1 bombillo, IN2 ventilador, IN3 cautín, IN4 plancha
RELES = (10, 11, 12, 13)
RELE_ACTIVO_BAJO = True

SAMPLE_RATE = 16000
BAUDIOS_CABLE = 921600  # modo cable (cable.py): el audio necesita ~320 000 baudios
PUERTO_AUDIO = 5000
PUERTO_CONTROL = 5001
NOMBRE_RED = "thonny"          # responde como thonny.local
# Potencia de transmisión WiFi en dBm. Con la potencia máxima la placa se reinicia
# ("Brownout detector was triggered") alimentada por el USB del PC.
WIFI_TXPOWER = 8
MAX_PLAY_BYTES = 16000 * 2 * 15  # máximo 15 s por orden "play"
