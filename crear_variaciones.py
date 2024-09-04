from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array
import os

input_dir = './imagenes/'
output_dir = './variaciones/'

os.makedirs(output_dir, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generar un número específico de variaciones por imagen
        num_variations = 640
        for i, batch in enumerate(
                datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='temp_', save_format='jpeg')):
            # Esperar que el archivo se guarde completamente antes de renombrar
            temp_filename = f'temp_{i + 1}.jpeg'
            new_filename = f'variacion{i + 1}.jpg'
            temp_filepath = os.path.join(output_dir, temp_filename)
            new_filepath = os.path.join(output_dir, new_filename)

            if os.path.exists(temp_filepath):
                # Cambiar el nombre de las imágenes generadas
                os.rename(temp_filepath, new_filepath)
            else:
                print(f'Archivo no encontrado: {temp_filepath}')

            if i >= num_variations - 1:
                break

        print(f'Variaciones generadas para {filename}')
