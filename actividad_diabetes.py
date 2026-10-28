from perceptron_comun import ejecutar


if __name__ == "__main__":
    ejecutar(
        archivo="Diabetes_SSEE.csv",
        caracteristicas=["IMC", "Glucosa"],
        objetivo="Diabetes",
        ejes=["IMC", "Glucosa"]
    )
