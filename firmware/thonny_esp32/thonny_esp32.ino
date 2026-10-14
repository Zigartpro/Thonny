// Firmware de Thonny para ESP32-S3 (núcleo Arduino ESP32 3.x).
//
// Funciona por WiFi (lo que usa Thonny) y por cable USB (pruebas y grabación de clips).
//
// WiFi, puerto 5000 (audio):   la ESP32 envía el audio del INMP441 sin parar,
//                              PCM 16 bits little-endian, 16 kHz, mono.
// WiFi, puerto 5001 (control): líneas JSON terminadas en '\n' (ver abajo).
// Cable USB (Serial, 921600 baudios): las mismas líneas JSON, más
//   {"cmd":"audio_serial"}   responde {"ok":true,...} y desde ahí manda solo audio crudo
//                            por el cable (igual que el puerto 5000) hasta que se reinicie.
//
// Órdenes JSON:
//   {"cmd":"rele","n":1,"on":true}     prende/apaga el relé n (1-4)
//   {"cmd":"todos_off"}                apaga los 4 relés
//   {"cmd":"play","bytes":N}           seguido de N bytes de PCM 16 kHz para el parlante
//   {"cmd":"estado"}                   estado de relés, WiFi y memoria
//   {"cmd":"ping"}
// Respuesta: una línea {"ok":true,...} o {"ok":false,"error":"..."}. Detalle en PROTOCOLO.md.

#include <WiFi.h>
#include <ESPmDNS.h>
#include <ESP_I2S.h>
#include "driver/gpio.h"
#include "config.h"
#include "secrets.h"

#define BAUDIOS 921600
#define WIFI_ESPERA_MS 20000

I2SClass i2sMic;
I2SClass i2sAmp;
WiFiServer servidorAudio(PUERTO_AUDIO);
WiFiServer servidorControl(PUERTO_CONTROL);
WiFiClient clienteAudio;
WiFiClient clienteControl;
bool estadoRele[4] = {false, false, false, false};
volatile bool audioPorSerial = false;   // en este modo el cable solo lleva audio
bool servidoresListos = false;
unsigned long ultimoIntentoWiFi = 0;

// Mensajes para el monitor serie (se callan cuando el cable lleva audio)
void registro(const char *fmt, ...) {
  if (audioPorSerial) return;
  char buf[200];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  Serial.print(buf);
}

// ------------------------------------------------------------------ relés

int nivelRele(bool encendido) {
  return (encendido ^ RELE_ACTIVO_BAJO) ? HIGH : LOW;
}

void relesApagadosAlArrancar() {
  // Primero se fija el nivel de "apagado" y solo después el pin pasa a ser salida,
  // para que ningún aparato se prenda solo al encender la placa.
  for (int i = 0; i < 4; i++) {
    gpio_set_level((gpio_num_t)RELES[i], nivelRele(false));
    pinMode(RELES[i], OUTPUT);
    digitalWrite(RELES[i], nivelRele(false));
  }
}

void ponerRele(int i, bool encendido) {
  digitalWrite(RELES[i], nivelRele(encendido));
  estadoRele[i] = encendido;
}

// ------------------------------------------------------------------ JSON mínimo

String valorTexto(const String &linea, const char *clave) {
  String k = String("\"") + clave + "\"";
  int p = linea.indexOf(k);
  if (p < 0) return "";
  p = linea.indexOf(':', p + k.length());
  int a = linea.indexOf('"', p + 1);
  int b = linea.indexOf('"', a + 1);
  return (a < 0 || b < 0) ? "" : linea.substring(a + 1, b);
}

long valorNumero(const String &linea, const char *clave, long porDefecto) {
  String k = String("\"") + clave + "\"";
  int p = linea.indexOf(k);
  if (p < 0) return porDefecto;
  p = linea.indexOf(':', p + k.length());
  return linea.substring(p + 1).toInt();
}

int valorBool(const String &linea, const char *clave) {  // 1, 0 o -1 si no está
  String k = String("\"") + clave + "\"";
  int p = linea.indexOf(k);
  if (p < 0) return -1;
  String resto = linea.substring(linea.indexOf(':', p + k.length()) + 1);
  resto.trim();
  if (resto.startsWith("true")) return 1;
  if (resto.startsWith("false")) return 0;
  return -1;
}

// ------------------------------------------------------------------ micrófono (tarea aparte)

void tareaMicrofono(void *) {
  const int N = 256;
  static int32_t crudo[N];
  static int16_t pcm[N];
  for (;;) {
    size_t leidos = i2sMic.readBytes((char *)crudo, sizeof(crudo));
    int n = leidos / 4;
    for (int i = 0; i < n; i++) {
      int32_t v = crudo[i] >> MIC_SHIFT;
      pcm[i] = (int16_t)constrain(v, -32768, 32767);
    }
    if (n <= 0) continue;
    if (audioPorSerial) {
      Serial.write((uint8_t *)pcm, n * 2);
    } else if (clienteAudio && clienteAudio.connected()) {
      clienteAudio.write((uint8_t *)pcm, n * 2);
    }
  }
}

// ------------------------------------------------------------------ órdenes (por WiFi o por cable)

void responder(Stream &canal, const String &json) {
  canal.print(json);
  canal.print('\n');
}

void reproducir(Stream &canal, long bytes) {
  if (bytes <= 0 || bytes > MAX_PLAY_BYTES || (bytes % 2) != 0) {
    responder(canal, "{\"ok\":false,\"error\":\"bytes invalido\"}");
    return;
  }
  static uint8_t buf[1024];
  long faltan = bytes;
  unsigned long ultimo = millis();
  while (faltan > 0) {
    int disponibles = canal.available();
    if (disponibles <= 0) {
      if (millis() - ultimo > 3000) break;  // el PC dejó de mandar
      delay(1);
      continue;
    }
    int n = canal.readBytes(buf, min((long)sizeof(buf), min((long)disponibles, faltan)));
    if (n > 0) {
      i2sAmp.write(buf, n);
      faltan -= n;
      ultimo = millis();
    }
  }
  if (faltan == 0) responder(canal, "{\"ok\":true}");
  else responder(canal, "{\"ok\":false,\"error\":\"audio incompleto\"}");
}

void atenderLinea(Stream &canal, String linea) {
  linea.trim();
  if (linea.length() == 0) return;
  String cmd = valorTexto(linea, "cmd");
  if (cmd == "ping") {
    responder(canal, "{\"ok\":true}");
  } else if (cmd == "rele") {
    long n = valorNumero(linea, "n", 0);
    int on = valorBool(linea, "on");
    if (n < 1 || n > 4 || on < 0) {
      responder(canal, "{\"ok\":false,\"error\":\"rele: n debe ser 1-4 y on true/false\"}");
      return;
    }
    ponerRele(n - 1, on == 1);
    responder(canal, "{\"ok\":true}");
  } else if (cmd == "todos_off") {
    for (int i = 0; i < 4; i++) ponerRele(i, false);
    responder(canal, "{\"ok\":true}");
  } else if (cmd == "play") {
    reproducir(canal, valorNumero(linea, "bytes", -1));
  } else if (cmd == "estado") {
    char s[240];
    snprintf(s, sizeof(s),
             "{\"ok\":true,\"reles\":[%d,%d,%d,%d],\"wifi\":%s,\"ip\":\"%s\",\"rssi\":%d,\"heap\":%u}",
             estadoRele[0], estadoRele[1], estadoRele[2], estadoRele[3],
             WiFi.status() == WL_CONNECTED ? "true" : "false", WiFi.localIP().toString().c_str(),
             WiFi.RSSI(), (unsigned)ESP.getFreeHeap());
    responder(canal, s);
  } else if (cmd == "audio_serial" && &canal == &Serial) {
    responder(canal, "{\"ok\":true,\"modo\":\"audio_serial\",\"sample_rate\":16000}");
    Serial.flush();
    audioPorSerial = true;
  } else {
    responder(canal, "{\"ok\":false,\"error\":\"comando desconocido\"}");
  }
}

// ------------------------------------------------------------------ WiFi

void iniciarWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // menos latencia en el audio
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  ultimoIntentoWiFi = millis();
  registro("Conectando a \"%s\"...\n", WIFI_SSID);
}

void revisarWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!servidoresListos) {
      registro("WiFi OK. IP: %s  RSSI: %d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
      if (MDNS.begin(NOMBRE_MDNS)) registro("Tambien como %s.local\n", NOMBRE_MDNS);
      servidorAudio.begin();
      servidorAudio.setNoDelay(true);
      servidorControl.begin();
      servidorControl.setNoDelay(true);
      registro("Audio en el puerto %d, control en el puerto %d\n", PUERTO_AUDIO, PUERTO_CONTROL);
      servidoresListos = true;
    }
    return;
  }
  if (millis() - ultimoIntentoWiFi > WIFI_ESPERA_MS) {  // reintenta sin bloquear el cable
    registro("WiFi sin conexion, reintentando (el cable USB sigue funcionando)\n");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    ultimoIntentoWiFi = millis();
  }
}

// ------------------------------------------------------------------ setup / loop

void setup() {
  relesApagadosAlArrancar();  // lo primero de todo
  Serial.setRxBufferSize(8192);
  Serial.begin(BAUDIOS);
  delay(200);
  registro("\n== Thonny ESP32-S3 ==\n");

  i2sMic.setPins(MIC_SCK, MIC_WS, -1, MIC_SD);
  if (!i2sMic.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT))
    registro("ERROR: no se pudo iniciar el I2S del microfono\n");
  i2sAmp.setPins(AMP_BCLK, AMP_LRC, AMP_DIN);
  if (!i2sAmp.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_BOTH))
    registro("ERROR: no se pudo iniciar el I2S del amplificador\n");

  iniciarWiFi();
  xTaskCreatePinnedToCore(tareaMicrofono, "mic", 8192, nullptr, 2, nullptr, 0);
  registro("Listo. Por cable: manda {\"cmd\":\"estado\"} a %d baudios\n", BAUDIOS);
}

void loop() {
  revisarWiFi();

  // Órdenes por cable (mientras no esté en modo audio)
  static String lineaSerial;
  while (!audioPorSerial && Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      atenderLinea(Serial, lineaSerial);
      lineaSerial = "";
    } else if (lineaSerial.length() < 256) {
      lineaSerial += c;
    }
  }

  // Órdenes por WiFi. Un cliente a la vez en cada puerto; uno nuevo reemplaza al anterior.
  if (servidoresListos) {
    if (servidorAudio.hasClient()) {
      if (clienteAudio) clienteAudio.stop();
      clienteAudio = servidorAudio.accept();
      clienteAudio.setNoDelay(true);
      registro("Audio: conectado %s\n", clienteAudio.remoteIP().toString().c_str());
    }
    if (servidorControl.hasClient()) {
      if (clienteControl) clienteControl.stop();
      clienteControl = servidorControl.accept();
      clienteControl.setNoDelay(true);
      registro("Control: conectado %s\n", clienteControl.remoteIP().toString().c_str());
    }
    static String lineaWiFi;
    while (clienteControl && clienteControl.connected() && clienteControl.available()) {
      char c = clienteControl.read();
      if (c == '\n') {
        atenderLinea(clienteControl, lineaWiFi);
        lineaWiFi = "";
      } else if (lineaWiFi.length() < 256) {
        lineaWiFi += c;
      }
    }
  }
  delay(2);
}
