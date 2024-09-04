import numpy as np
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array

# Parámetros
longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.weights.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x = load_img(file, target_size=(longitud, altura), color_mode='grayscale')
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)  # Añadir una dimensión para el batch
    arreglo = cnn.predict(x)  # Predecir la clase
    resultado = np.squeeze(arreglo)  # Convertir el arreglo a un vector plano

    # Definir las clases según la lógica
    clases = {
        0: "Billete 10 (División Antigua) - Anverso",
        1: "Billete 10 (División Nueva) - Anverso",
        2: "Billete 10 (División Antigua) - Reverso",
        3: "Billete 10 (División Nueva) - Reverso",
        4: "Billete 20 (División Antigua) - Anverso",
        5: "Billete 20 (División Nueva) - Anverso",
        6: "Billete 20 (División Antigua) - Reverso",
        7: "Billete 20 (División Nueva) - Reverso",
        8: "Billete 50 (División Antigua) - Anverso",
        9: "Billete 50 (División Nueva) - Anverso",
        10: "Billete 50 (División Antigua) - Reverso",
        11: "Billete 50 (División Nueva) - Reverso",
        12: "Billete 100 (División Antigua) - Anverso",
        13: "Billete 100 (División Nueva) - Anverso",
        14: "Billete 100 (División Antigua) - Reverso",
        15: "Billete 100 (División Nueva) - Reverso",
        16: "Billete 200 (División Antigua) - Anverso",
        17: "Billete 200 (División Nueva) - Anverso",
        18: "Billete 200 (División Antigua) - Reverso",
        19: "Billete 200 (División Nueva) - Reverso",
    }

    # Obtener el índice de la clase con mayor probabilidad
    respuesta = np.argmax(resultado)
    probabilidad = resultado[respuesta] * 100  # Obtener el porcentaje de confianza para la clase elegida

    # Imprimir el porcentaje de confianza y la predicción
    print(f'Predicción: {clases[respuesta]} con una confianza de {probabilidad:.2f}%')

    return respuesta

# Ejemplo de uso
predict('./Dataset/test/billete100-224.jpeg')