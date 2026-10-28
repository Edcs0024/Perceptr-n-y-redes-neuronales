from perceptron_comun import ejecutar


if __name__ == "__main__":
    ejecutar(
        archivo="Anemia.csv",
        caracteristicas=["Edad", "Energia", "Frecuencia", "Antecedentes"],
        objetivo="Anemia",
        ejes=["Edad", "Frecuencia"]
    )
