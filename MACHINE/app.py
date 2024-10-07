from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Ruta donde están las carpetas con imágenes
carpeta_modelo = 'C:/Users/nob96/OneDrive/Escritorio/eva1 MACHINE/modelo'
model_path = os.path.join(carpeta_modelo, 'modelo.h5')

# Cargar el modelo
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Etiquetas de las clases
class_labels = ['elefante', 'farfalla', 'leon']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    
    if not file:
        return jsonify({'error': 'No se ha enviado ninguna imagen.'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')  # Asegúrate de que la imagen sea RGB
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realiza la predicción
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return jsonify({'class': class_labels[predicted_class], 'probabilities': predictions.tolist()})
    
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return jsonify({'error': str(e)}), 500

def cargar_imagenes(carpeta):
    imagenes = []
    etiquetas = []

    for clase in class_labels:
        ruta_clase = os.path.join(carpeta, f"{clase}_train")
        if not os.path.exists(ruta_clase):
            print(f'Ruta no encontrada: {ruta_clase}')
            continue
            
        for nombre_archivo in os.listdir(ruta_clase):
            if nombre_archivo.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(ruta_clase, nombre_archivo)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')  # Asegúrate de que sea RGB
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        
                        if img_array.shape == (224, 224, 3):
                            imagenes.append(img_array)
                            etiquetas.append(class_labels.index(clase))
                        else:
                            print(f'Imagen {nombre_archivo} no tiene dimensiones válidas: {img_array.shape}')
                
                except Exception as e:
                    print(f'Error al procesar la imagen {nombre_archivo}: {e}')
    
    return np.array(imagenes), np.array(etiquetas)

# Cargar imágenes al iniciar la aplicación
imagenes_cargadas, etiquetas_cargadas = cargar_imagenes(carpeta_modelo)

# Dividir datos en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(imagenes_cargadas, etiquetas_cargadas, test_size=0.2, random_state=42)

# Aumentación de datos
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Transfer Learning: usar MobileNetV2 como base
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar las capas

# Construir el modelo
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con la aumentación de datos
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          epochs=10, 
          validation_data=(X_val, y_val))

# Guardar el modelo entrenado
model.save(model_path)

if __name__ == '__main__':
    app.run(debug=True)
