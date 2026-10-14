# Comparación de experimentos de reconocimiento de hablante

Generado por `pc/evaluar.py`. Umbral escogido en `val`; todas las métricas sobre `test` (Audio 5 de cristian y johan, clips apartados de rocko, y 6 hablantes públicos que el modelo nunca vio).

| Exp. | Clasificador | Clips de miembros en train | Umbral | Aceptación correcta (miembros) | Identificación | Falsa aceptación (ajenos) | EER test |
|---|---|---|---|---|---|---|---|
| E1 | coseno | 3 | 0.223 | 97.8 % | 100.0 % | ajeno_publico: 0.0 % / ajeno_sintetico: 4.2 % | 2.2 % |
| E1 | reglog | 3 | 0.083 | 100.0 % | 100.0 % | ajeno_publico: 0.0 % / ajeno_sintetico: 2.1 % | 0.2 % |
| E2 | coseno | 115 | 0.393 | 93.3 % | 100.0 % | ajeno_publico: 0.0 % / ajeno_sintetico: 0.0 % | 0.0 % |
| E2 | reglog | 115 | 0.521 | 97.8 % | 100.0 % | ajeno_publico: 0.0 % / ajeno_sintetico: 0.0 % | 0.0 % |
| E4 | coseno | 460 | 0.365 | 93.3 % | 100.0 % | ajeno_publico: 0.0 % / ajeno_sintetico: 0.0 % | 0.2 % |
| E4 | reglog | 460 | 0.641 | 95.6 % | 100.0 % | ajeno_publico: 0.0 % / ajeno_sintetico: 0.0 % | 0.0 % |
