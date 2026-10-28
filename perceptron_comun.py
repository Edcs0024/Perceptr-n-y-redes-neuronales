
import time
import pandas as pd
import matplotlib.pyplot as plt
import serial

PUERTO = "COM5"
BAUDIOS = 115200
EPOCAS = 50


def ejecutar(archivo, caracteristicas, objetivo, ejes):
    df = pd.read_csv(archivo)
    df.columns = df.columns.str.strip()

    columnas = caracteristicas + [objetivo]
    df = df[columnas].dropna().copy()

    for col in columnas:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.empty:
        raise ValueError(f"No hay registros validos en {archivo} para las columnas {columnas}.")

    if not df[objetivo].isin([0, 1]).all():
        raise ValueError(f"{objetivo} debe contener solamente 0 y 1.")

    if len(caracteristicas) > 6:
        raise ValueError("El Arduino UNO admite como maximo 6 caracteristicas.")

    X = df[caracteristicas].astype(float)
    y = df[objetivo].astype(int)

    medias = X.mean()
    escalas = X.std(ddof=0).replace(0, 1)
    Xn = (X - medias) / escalas

    # Grafica de los datos originales
    plt.figure(figsize=(8, 6))

    for clase, etiqueta, marcador in [
        (0, "Clase 0", "o"),
        (1, "Clase 1", "x")
    ]:
        mascara = y == clase
        plt.scatter(
            df.loc[mascara, ejes[0]],
            df.loc[mascara, ejes[1]],
            label=etiqueta,
            marker=marcador
        )

    plt.xlabel(ejes[0])
    plt.ylabel(ejes[1])
    plt.title(f"Datos originales: {objetivo}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{objetivo}_dispersion.png", dpi=200)
    plt.show()

    salida = {}

    # Comunicacion con Arduino UNO
    print(f"Conectando con Arduino en {PUERTO} a {BAUDIOS} baudios...")
    print("Cierra el Monitor Serie del Arduino IDE antes de continuar.")
    with serial.Serial(PUERTO, BAUDIOS, timeout=30) as arduino:
        time.sleep(2)
        arduino.reset_input_buffer()

        arduino.write(f"N={len(df)}\n".encode())
        arduino.write(f"F={len(caracteristicas)}\n".encode())
        arduino.write(f"E={EPOCAS}\n".encode())

        respuesta = arduino.readline().decode(
            errors="ignore"
        ).strip()

        if respuesta != "READY":
            raise RuntimeError(
                f"Se esperaba READY, pero Arduino respondio: {respuesta}"
            )

        print("Arduino listo. Iniciando entrenamiento...")

        # Enviar un registro y esperar ACK antes del siguiente
        for epoca in range(EPOCAS):
            print(f"Epoca {epoca + 1}/{EPOCAS}")

            for i in range(len(df)):
                valores = Xn.iloc[i].tolist() + [int(y.iloc[i])]
                mensaje = ",".join(
                    f"{v:.6f}" for v in valores
                ) + "\n"

                arduino.write(mensaje.encode())

                respuesta = arduino.readline().decode(
                    errors="ignore"
                ).strip()

                if respuesta != "ACK":
                    raise RuntimeError(
                        f"Fallo en epoca {epoca + 1}, "
                        f"registro {i + 1}. "
                        f"Respuesta: {respuesta}"
                    )

        print("Entrenamiento enviado. Recibiendo resultados...")

        while True:
            linea = arduino.readline().decode(
                errors="ignore"
            ).strip()

            if not linea:
                raise TimeoutError(
                    "Arduino no envio los resultados. "
                    "Comprueba que el sketch responda ACK por registro."
                )

            print("Arduino:", linea)

            if linea == "DONE":
                break

            if "=" in linea:
                clave, valor = linea.split("=", 1)
                try:
                    salida[clave] = float(valor)
                except ValueError:
                    pass

    # Comprobar que Arduino envio todos los pesos
    claves_pesos = [f"W{i}" for i in range(len(caracteristicas))]

    faltantes = [
        clave for clave in claves_pesos + ["BIAS"]
        if clave not in salida
    ]

    if faltantes:
        raise RuntimeError(
            f"Faltan resultados de Arduino: {faltantes}"
        )

    pesos = [salida[clave] for clave in claves_pesos]
    bias = salida["BIAS"]

    # Evaluacion en Python
    puntuaciones = Xn.to_numpy() @ pesos + bias
    predicciones = (puntuaciones >= 0).astype(int)
    exactitud = (predicciones == y.to_numpy()).mean() * 100

    print("\n========== RESULTADOS ==========")
    print("Dataset:", archivo)
    print("Registros:", len(df))
    print("Pesos:", pesos)
    print("Bias:", bias)
    print("Epocas:", EPOCAS)
    print(f"Exactitud: {exactitud:.2f}%")

    # Guardar predicciones
    resultado = df.copy()
    resultado["Real"] = y.to_numpy()
    resultado["Prediccion"] = predicciones
    resultado.to_csv(f"{objetivo}_resultados.csv", index=False)

    # Grafica de frontera de decision
    if len(caracteristicas) == 2 and ejes == caracteristicas:
        w0, w1 = pesos

        if abs(w1) > 1e-10:
            xline = [
                df[ejes[0]].min() - 1,
                df[ejes[0]].max() + 1
            ]

            x_normalizado = (
                pd.Series(xline) - medias[ejes[0]]
            ) / escalas[ejes[0]]

            y_normalizado = (
                -w0 * x_normalizado - bias
            ) / w1

            yline = (
                y_normalizado * escalas[ejes[1]]
                + medias[ejes[1]]
            )

            plt.figure(figsize=(8, 6))

            for clase, etiqueta, marcador in [
                (0, "Clase 0", "o"),
                (1, "Clase 1", "x")
            ]:
                mascara = y == clase
                plt.scatter(
                    df.loc[mascara, ejes[0]],
                    df.loc[mascara, ejes[1]],
                    label=etiqueta,
                    marker=marcador
                )

            plt.plot(
                xline, yline,
                label="Frontera del perceptron"
            )
            plt.xlabel(ejes[0])
            plt.ylabel(ejes[1])
            plt.title(f"Frontera de decision: {objetivo}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{objetivo}_frontera.png", dpi=200)
            plt.show()

    # Prueba con nuevos datos
    print("\nPrueba con una nueva entrada.")
    print("Escribe salir para terminar.")

    while True:
        nuevos = []

        for nombre in caracteristicas:
            entrada = input(f"{nombre}: ").strip()

            if entrada.lower() == "salir":
                return

            try:
                nuevos.append(float(entrada))
            except ValueError:
                print("Valor no valido. Intenta de nuevo.")
                break
        else:
            nuevos_n = [
                (v - medias[nombre]) / escalas[nombre]
                for v, nombre in zip(nuevos, caracteristicas)
            ]

            suma = sum(
                w * v for w, v in zip(pesos, nuevos_n)
            ) + bias

            print("Prediccion del perceptron:", int(suma >= 0))
            print("Resultado experimental, no diagnostico medico.")
