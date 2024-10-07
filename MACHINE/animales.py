import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score,
                            ConfusionMatrixDisplay)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                    Dense, Dropout)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear una ventana para seleccionar la carpeta
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal

# Abrir un diálogo para seleccionar la carpeta que contiene las imágenes
folder_path = filedialog.askdirectory(title="Selecciona la carpeta que contiene las imágenes")

# Diccionario para mapear nombres de carpetas a etiquetas
class_names = {
    'elefante_train': 0,
    'farfalla_train': 1,
    'mucca_train': 2,
    'pecora_train': 3,
    'scoiattolo_train': 4
}

# Listas para almacenar imágenes y etiquetas
image_list = []
y = []

# Recorrer cada carpeta y sus archivos
for class_name, label in class_names.items():
    class_folder = os.path.join(folder_path, class_name)
    if os.path.exists(class_folder):
        for filename in os.listdir(class_folder):
            if filename.lower().endswith((".jpg", ".png")):
                try:
                    img_path = os.path.join(class_folder, filename)
                    img = Image.open(img_path)
                    img = img.resize((224, 224))  # Redimensionar la imagen
                    img_array = np.array(img) / 255.0  # Normalizar valores de píxeles (0-1)

                    # Verificar si la imagen tiene el tamaño correcto
                    if img_array.shape == (224, 224, 3):
                        image_list.append(img_array)
                        y.append(label)  # Añadir la etiqueta correspondiente
                    else:
                        print(f"Imagen {filename} no tiene el tamaño esperado.")
                except Exception as e:
                    print(f"Error al cargar {filename}: {e}")
    else:
        print(f"Carpeta no encontrada: {class_folder}")

# Convertir listas a arrays de numpy
image_list = np.array(image_list)
y = np.array(y)

# Imprimir conteos
print(f"Total de imágenes cargadas: {len(image_list)}")
print(f"Total de etiquetas: {len(y)}")

# Verificar que las longitudes coincidan
if len(image_list) != len(y):
    print("Error: el número de imágenes y etiquetas no coincide.")
    exit(1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    image_list, y, test_size=0.2, random_state=42, stratify=y)

# Configurar la augmentación de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Intentar cargar el modelo entrenado
try:
    model = load_model('mi_modelo.h5')  # Intentar cargar el modelo
    print("Modelo cargado exitosamente.")
except:
    print("No se encontró el modelo, se creará uno nuevo.")
    # Configurar el modelo de red neuronal
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Añadir Dropout para regularización
        Dense(len(class_names), activation='softmax')  # Cambiar al número de clases
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Ajustar el modelo usando la augmentación de datos
    model.fit(datagen.flow(X_train, y_train, batch_size=32), 
            epochs=20, 
            validation_data=(X_test, y_test))

    # Guardar el modelo entrenado
    model.save('mi_modelo.h5')  # Guardar el modelo en un archivo HDF5

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Hacer predicciones
y_pred = model.predict(X_test)

# Convertir predicciones a etiquetas de clase
y_pred = np.argmax(y_pred, axis=1)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Obtener las clases únicas en y_test y y_pred
unique_classes = np.unique(np.concatenate((y_test, y_pred)))
print("Clases únicas en y_test y y_pred:", unique_classes)

# Definir las etiquetas de clase correspondientes
class_labels = ['elefante', 'farfalla', 'mucca', 'pecora', 'scoiattolo']

# Limitar las etiquetas a las que realmente están presentes
limited_labels = [class_labels[i] for i in unique_classes]

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=limited_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.show()  # Mostrar el gráfico

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None, zero_division=0)
precision_avg = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average=None, zero_division=0)
recall_avg = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
f1_avg = f1_score(y_test, y_pred, average="macro", zero_division=0)

# Mostrar una imagen de prueba
img_index = np.random.randint(0, len(X_test))  # Seleccionar un índice aleatorio
if img_index < len(X_test):
    plt.imshow(X_test[img_index])  # Mostrar la imagen
    plt.title(f"Predicción: {y_pred[img_index]}, Real: {y_test[img_index]}")
    plt.axis('off')  # No mostrar los ejes
    plt.show()
else:
    print("Índice de imagen fuera de rango.")
