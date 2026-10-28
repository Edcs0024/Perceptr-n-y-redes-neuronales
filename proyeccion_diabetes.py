"""
Actividad: Proyección de diabetes con perceptrón simple
------------------------------------------------------
Este programa:
1. Lee el archivo de datos (.xlsx o .csv)
2. Grafica IMC vs Glucosa
3. Envía los datos al Arduino
4. Recibe los pesos entrenados por el Arduino
5. Calcula la exactitud
6. Muestra la ecuación de la recta de separación
7. Grafica la frontera de decisión

Instalación de librerías:
    pip install pandas matplotlib openpyxl pyserial
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import serial
from serial.tools import list_ports

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

ARCHIVO = "Diabetes_SSEE.csv"
BAUDIOS = 9600

# Si quieres poner el puerto manualmente, escribe por ejemplo:
# PUERTO = "COM3"
# Para intentar detectarlo automáticamente:
PUERTO = None


# ---------------------------------------------------------
# 1. CARGAR DATOS
# ---------------------------------------------------------

def cargar_datos(archivo):
    if archivo.lower().endswith(".csv"):
        df = pd.read_csv(archivo)
    elif archivo.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(archivo)
    else:
        raise ValueError("El archivo debe ser CSV o Excel.")

    # Limpiar nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]

    columnas_necesarias = ["IMC", "Glucosa", "Diabetes"]

    for columna in columnas_necesarias:
        if columna not in df.columns:
            raise ValueError(
                f"No se encontró la columna '{columna}'. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    # Conservar únicamente las columnas necesarias y quitar filas vacías
    df = df[columnas_necesarias].dropna()

    df["IMC"] = pd.to_numeric(df["IMC"])
    df["Glucosa"] = pd.to_numeric(df["Glucosa"])
    df["Diabetes"] = pd.to_numeric(df["Diabetes"]).astype(int)

    return df


# ---------------------------------------------------------
# 2. GRÁFICA DE DISPERSIÓN
# ---------------------------------------------------------

def graficar_datos(df):
    no_diabetes = df[df["Diabetes"] == 0]
    si_diabetes = df[df["Diabetes"] == 1]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        no_diabetes["IMC"],
        no_diabetes["Glucosa"],
        label="Sin diabetes (0)",
        marker="o"
    )

    plt.scatter(
        si_diabetes["IMC"],
        si_diabetes["Glucosa"],
        label="Con diabetes (1)",
        marker="x"
    )

    plt.xlabel("IMC")
    plt.ylabel("Glucosa (mg/dL)")
    plt.title("Datos originales: IMC vs Glucosa")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("01_dispersion_diabetes.png", dpi=200)
    plt.show()


# ---------------------------------------------------------
# 3. DETECTAR ARDUINO
# ---------------------------------------------------------

def detectar_puerto():
    puertos = list(list_ports.comports())

    if len(puertos) == 0:
        raise RuntimeError(
            "No se detectó ningún puerto serial. "
            "Conecta el Arduino y vuelve a intentar."
        )

    print("\nPuertos detectados:")
    for i, puerto in enumerate(puertos):
        print(f"{i}: {puerto.device} - {puerto.description}")

    if len(puertos) == 1:
        print(f"\nSe usará automáticamente: {puertos[0].device}")
        return puertos[0].device

    indice = int(input("\nEscribe el número del puerto del Arduino: "))
    return puertos[indice].device


# ---------------------------------------------------------
# 4. ENVIAR DATOS AL ARDUINO Y RECIBIR PESOS
# ---------------------------------------------------------

def entrenar_en_arduino(df, puerto):
    print(f"\nConectando con Arduino en {puerto}...")

    arduino = serial.Serial(puerto, BAUDIOS, timeout=2)

    # Muchos Arduino se reinician al abrir el puerto serial.
    time.sleep(2.5)
    arduino.reset_input_buffer()

    print("Enviando datos al Arduino...")

    # Primero mandamos la cantidad de registros
    arduino.write(f"N={len(df)}\n".encode("utf-8"))
    time.sleep(0.2)

    # Después enviamos cada fila: IMC,Glucosa,Diabetes
    for _, fila in df.iterrows():
        mensaje = (
            f"{fila['IMC']:.4f},"
            f"{fila['Glucosa']:.4f},"
            f"{int(fila['Diabetes'])}\n"
        )
        arduino.write(mensaje.encode("utf-8"))
        time.sleep(0.01)

    print("Datos enviados. Esperando entrenamiento...\n")

    w1 = None
    w2 = None
    bias = None
    epocas = None
    exactitud_arduino = None

    tiempo_inicio = time.time()

    while time.time() - tiempo_inicio < 15:
        if arduino.in_waiting:
            linea = arduino.readline().decode("utf-8", errors="ignore").strip()

            if not linea:
                continue

            print("Arduino:", linea)

            if linea.startswith("W1="):
                w1 = float(linea.split("=")[1])

            elif linea.startswith("W2="):
                w2 = float(linea.split("=")[1])

            elif linea.startswith("BIAS="):
                bias = float(linea.split("=")[1])

            elif linea.startswith("EPOCAS="):
                epocas = int(linea.split("=")[1])

            elif linea.startswith("EXACTITUD="):
                exactitud_arduino = float(linea.split("=")[1])

            elif linea == "LISTO_PARA_OTRO_ENTRENAMIENTO":
                break

    arduino.close()

    if w1 is None or w2 is None or bias is None:
        raise RuntimeError(
            "No se recibieron correctamente los pesos del Arduino."
        )

    return w1, w2, bias, epocas, exactitud_arduino


# ---------------------------------------------------------
# 5. EVALUAR EL MODELO EN PYTHON
# ---------------------------------------------------------

def evaluar(df, w1, w2, bias):
    predicciones = []

    for _, fila in df.iterrows():
        suma = (
            w1 * fila["IMC"]
            + w2 * fila["Glucosa"]
            + bias
        )

        prediccion = 1 if suma >= 0 else 0
        predicciones.append(prediccion)

    df = df.copy()
    df["Prediccion"] = predicciones

    correctos = (df["Prediccion"] == df["Diabetes"]).sum()
    exactitud = correctos / len(df) * 100

    return df, exactitud


# ---------------------------------------------------------
# 6. ECUACIÓN Y FRONTERA DE DECISIÓN
# ---------------------------------------------------------

def mostrar_ecuacion(w1, w2, bias):
    print("\n" + "=" * 60)
    print("ECUACIÓN DEL PERCEPTRÓN")
    print("=" * 60)

    print(
        f"{w1:.6f}(IMC) + "
        f"{w2:.6f}(Glucosa) + "
        f"{bias:.6f} = 0"
    )

    if w2 != 0:
        pendiente = -(w1 / w2)
        intercepto = -(bias / w2)

        print("\nDespejando Glucosa:")
        print(
            f"Glucosa = {pendiente:.6f}(IMC) "
            f"+ {intercepto:.6f}"
        )

        return pendiente, intercepto

    return None, None


def graficar_frontera(df, pendiente, intercepto):
    no_diabetes = df[df["Diabetes"] == 0]
    si_diabetes = df[df["Diabetes"] == 1]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        no_diabetes["IMC"],
        no_diabetes["Glucosa"],
        label="Sin diabetes (0)",
        marker="o"
    )

    plt.scatter(
        si_diabetes["IMC"],
        si_diabetes["Glucosa"],
        label="Con diabetes (1)",
        marker="x"
    )

    xmin = df["IMC"].min() - 1
    xmax = df["IMC"].max() + 1

    x1 = xmin
    x2 = xmax

    y1 = pendiente * x1 + intercepto
    y2 = pendiente * x2 + intercepto

    plt.plot(
        [x1, x2],
        [y1, y2],
        label="Frontera del perceptrón"
    )

    plt.xlabel("IMC")
    plt.ylabel("Glucosa (mg/dL)")
    plt.title("Perceptrón: frontera de separación")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("02_frontera_perceptron.png", dpi=200)
    plt.show()


# ---------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------

def main():
    print("Cargando dataset...")
    df = cargar_datos(ARCHIVO)

    print("\nPrimeras filas:")
    print(df.head())

    print(f"\nCantidad de registros: {len(df)}")
    print(f"Sin diabetes: {(df['Diabetes'] == 0).sum()}")
    print(f"Con diabetes: {(df['Diabetes'] == 1).sum()}")

    # Gráfica original
    graficar_datos(df)

    # Puerto Arduino
    puerto = PUERTO if PUERTO is not None else detectar_puerto()

    # Entrenamiento en Arduino
    w1, w2, bias, epocas, exactitud_arduino = entrenar_en_arduino(
        df, puerto
    )

    # Evaluación en Python
    resultado, exactitud_python = evaluar(df, w1, w2, bias)

    pendiente, intercepto = mostrar_ecuacion(w1, w2, bias)

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    print(f"W1: {w1:.6f}")
    print(f"W2: {w2:.6f}")
    print(f"Bias: {bias:.6f}")
    print(f"Épocas: {epocas}")
    print(f"Exactitud calculada por Arduino: {exactitud_arduino:.2f}%")
    print(f"Exactitud calculada por Python: {exactitud_python:.2f}%")

    # Mostrar algunos resultados
    print("\nPredicciones:")
    print(resultado.head(15))

    if pendiente is not None:
        graficar_frontera(resultado, pendiente, intercepto)

    print("\nArchivos de evidencia generados:")
    print("  01_dispersion_diabetes.png")
    print("  02_frontera_perceptron.png")


if __name__ == "__main__":
    main()
