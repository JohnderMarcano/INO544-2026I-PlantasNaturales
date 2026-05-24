import tensorflow as tf
from tensorflow.keras import layers, models
import tf2onnx
import matplotlib.pyplot as plt
import os

# 1. RUTA 
ruta_dataset = './procesadas'
tamaño_lote = 32

print("Fase 1: Cargando imágenes y dividiendo 80/20...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
  ruta_dataset,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=tamaño_lote)

test_dataset = tf.keras.utils.image_dataset_from_directory(
  ruta_dataset,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=tamaño_lote)

# Optimización de carga
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("\nFase 2: Construyendo la CNN...")
# Input Shape: [1, 224, 224, 3] y nombre "cam_input"
entrada = layers.Input(shape=(224, 224, 3), name="cam_input", dtype=tf.float32)

x = layers.Rescaling(1./255)(entrada) 
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

# Output Shape: [1, 1] - Activación Salida: Sigmoid
salida = layers.Dense(1, activation='sigmoid', name="confidence_score")(x)

modelo = models.Model(inputs=entrada, outputs=salida)
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nFase 3: ¡Entrenando la red neuronal!...")
historial = modelo.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=10 
)

print("\nFase 4: Exportando a ONNX v12+...")

os.makedirs('model', exist_ok=True)

# Especificaciones para ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="cam_input"),)
output_path = "./model/modelo_plantas.onnx"

tf2onnx.convert.from_keras(modelo, input_signature=spec, opset=13, output_path=output_path)

print(f"\n¡CORONAMOS! Modelo guardado exitosamente en: {output_path}")

# --- FASE 5: GENERACIÓN DE GRÁFICAS 
print("\nFase 5: Generando gráfica de rendimiento para el README...")
os.makedirs('src', exist_ok=True) 

plt.figure(figsize=(12, 5))

# Gráfica de Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Entrenamiento', color='green')
plt.plot(historial.history['val_accuracy'], label='Validación (Test)', color='blue', linestyle='--')
plt.title('Curva de Precisión (Accuracy)')
plt.xlabel('Épocas (Epochs)')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Entrenamiento', color='red')
plt.plot(historial.history['val_loss'], label='Validación (Test)', color='orange', linestyle='--')
plt.title('Curva de Pérdida (Loss / Error)')
plt.xlabel('Épocas (Epochs)')
plt.ylabel('Pérdida (Binary Crossentropy)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Guardar la imagen en la carpeta src
plt.tight_layout()
plt.savefig('src/grafica_rendimiento.png', dpi=300)
print("✅ ¡Gráfica guardada exitosamente en src/grafica_rendimiento.png!")