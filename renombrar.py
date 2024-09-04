import os

image_dir = './Dataset/Billete200-trasera-2/'
prefix = 'billete200-'

# Obtener todos los archivos en el directorio que terminan en .jpg, .jpeg o .png
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Ordenar los archivos
images = sorted(images)

# Renombrar archivos
for i, filename in enumerate(images):
    ext = os.path.splitext(filename)[1]
    new_name = f"{prefix}{i + 1}{ext}"
    old_path = os.path.join(image_dir, filename)
    new_path = os.path.join(image_dir, new_name)

    # Comprobar si el nuevo nombre ya existe
    if not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f'Renombrado: {filename} -> {new_name}')
    else:
        print(f'El archivo {new_name} ya existe. Ignorando.')
