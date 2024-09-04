from PIL import Image
import os

input_dir = './imagenes/'
output_dir = './imagenesoptimizadas/'

os.makedirs(output_dir, exist_ok=True)

new_width = 500
new_height = 500


for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        output_path = os.path.join(output_dir, filename)
        img.save(output_path, optimize=True, quality=85)
        print(f'{filename} ha sido redimensionada y guardada en {output_path}')
