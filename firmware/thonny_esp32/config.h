// Configuración del firmware de Thonny (ESP32-S3-N16R8).
// Pines elegidos según las reglas del plan: sin GPIO 0, 3, 19-20, 26-37, 45 ni 46.
#pragma once

// ---------------- Micrófono INMP441 (I2S de entrada) ----------------
// VDD -> 3V3, GND -> GND, L/R -> GND (canal izquierdo)
#define MIC_SCK 15
#define MIC_WS  16
#define MIC_SD  17
// El INMP441 entrega 24 bits dentro de palabras de 32. Se desplaza a 16 bits;
// un desplazamiento menor = más ganancia (14 => +12 dB respecto a 16).
#define MIC_SHIFT 14

// ---------------- Amplificador MAX98357 (I2S de salida) ----------------
// VIN -> 5V, GND -> GND, SD y GAIN sin conectar
#define AMP_BCLK 5
#define AMP_LRC  6
#define AMP_DIN  7

// ---------------- Relés (módulo de 4 canales) ----------------
// Jumper VCC-JD-VCC quitado: JD-VCC -> 5V, VCC -> 3V3, GND -> GND
#define RELE_ACTIVO_BAJO true     // el módulo cierra el relé con IN en 0 V
static const int RELES[4] = {10, 11, 12, 13};  // IN1 bombillo, IN2 ventilador, IN3 cautín, IN4 plancha

// ---------------- Audio y red ----------------
#define SAMPLE_RATE   16000
#define PUERTO_AUDIO  5000   // ESP32 -> PC: PCM 16 bits LE, 16 kHz, mono, continuo
#define PUERTO_CONTROL 5001  // PC -> ESP32: líneas JSON + audio para el parlante
#define NOMBRE_MDNS   "thonny"   // la placa responde como thonny.local
#define MAX_PLAY_BYTES (16000 * 2 * 15)  // máximo 15 s de audio por orden "play"
