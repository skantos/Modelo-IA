
# pip install tensorflow tensorflowjs
# pip install --upgrade tensorflowjs


import tensorflow as tf
import tensorflowjs as tfjs

# Carga tu modelo Keras
model = tf.keras.models.load_model('./entrenamiento/modelo.h5')

# Convierte y guarda el modelo en formato TensorFlow.js
tfjs.converters.save_keras_model(model, './entrenamiento/model_tfjs')  # El modelo se guardará en 'model_tfjs' carpeta
