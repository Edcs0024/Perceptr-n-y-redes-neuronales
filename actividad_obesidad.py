from perceptron_comun import ejecutar


if __name__ == "__main__":
    ejecutar(
        archivo="Obesidad.csv",
        caracteristicas=["Edad", "Sexo", "IMC", "Actividad", "Calorias", "Antecedentes"],
        objetivo="Obesidad",
        ejes=["Edad", "IMC"]
    )
