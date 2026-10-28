
#include <string.h>
#include <stdlib.h>

/*
  Perceptron para Arduino UNO.
  Python envia los registros en cada epoca.
  Protocolo: N, F, E -> READY -> filas/ACK -> resultados -> DONE
*/

const int MAX_CARACTERISTICAS = 6;
const int MAX_EPOCAS = 100;

float pesos[MAX_CARACTERISTICAS];
float bias = 0.0;

int nDatos = 0;
int nCaracteristicas = 0;
int nEpocas = 0;

char linea[160];

bool leerLinea() {
  size_t n = Serial.readBytesUntil('\n', linea, sizeof(linea) - 1);
  linea[n] = '\0';

  while (n > 0 &&
         (linea[n - 1] == '\r' || linea[n - 1] == ' ')) {
    linea[--n] = '\0';
  }

  return n > 0;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10000);
  Serial.println("ARDUINO_LISTO");
}

void loop() {
  if (!Serial.available()) return;

  if (!leerLinea()) return;

  if (strncmp(linea, "N=", 2) != 0) {
    Serial.println("ERROR_N");
    return;
  }
  nDatos = atoi(linea + 2);

  if (!leerLinea() || strncmp(linea, "F=", 2) != 0) {
    Serial.println("ERROR_F");
    return;
  }
  nCaracteristicas = atoi(linea + 2);

  if (!leerLinea() || strncmp(linea, "E=", 2) != 0) {
    Serial.println("ERROR_E");
    return;
  }
  nEpocas = atoi(linea + 2);

  if (nDatos < 1 || nDatos > 300 ||
      nCaracteristicas < 1 ||
      nCaracteristicas > MAX_CARACTERISTICAS ||
      nEpocas < 1 || nEpocas > MAX_EPOCAS) {
    Serial.println("ERROR_CONFIG");
    return;
  }

  for (int j = 0; j < MAX_CARACTERISTICAS; j++) {
    pesos[j] = 0.0;
  }
  bias = 0.0;

  Serial.println("READY");

  int errores = 0;

  for (int epoca = 0; epoca < nEpocas; epoca++) {
    errores = 0;

    for (int i = 0; i < nDatos; i++) {
      if (!leerLinea()) {
        Serial.println("ERROR_FILA");
        return;
      }

      char *token = strtok(linea, ",");
      float valores[MAX_CARACTERISTICAS + 1];

      for (int j = 0; j <= nCaracteristicas; j++) {
        if (token == NULL) {
          Serial.println("ERROR_COLUMNAS");
          return;
        }

        valores[j] = atof(token);
        token = strtok(NULL, ",");
      }

      float suma = bias;

      for (int j = 0; j < nCaracteristicas; j++) {
        suma += pesos[j] * valores[j];
      }

      int prediccion = (suma >= 0.0) ? 1 : 0;
      int esperado = (valores[nCaracteristicas] >= 0.5) ? 1 : 0;
      int error = esperado - prediccion;

      if (error != 0) {
        for (int j = 0; j < nCaracteristicas; j++) {
          pesos[j] += 0.01 * error * valores[j];
        }

        bias += 0.01 * error;
        errores++;
      }

      // Confirmar que esta fila ya fue procesada
      Serial.println("ACK");
    }
  }

  for (int j = 0; j < nCaracteristicas; j++) {
    Serial.print("W");
    Serial.print(j);
    Serial.print("=");
    Serial.println(pesos[j], 8);
  }

  Serial.print("BIAS=");
  Serial.println(bias, 8);

  Serial.print("EPOCHS=");
  Serial.println(nEpocas);

  Serial.print("LAST_ERRORS=");
  Serial.println(errores);

  Serial.println("DONE");
}
