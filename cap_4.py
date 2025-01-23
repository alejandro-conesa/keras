import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# print('Tamaño de los datos de entrenamiento:')
# print(f' - Imágenes: {X_train.shape}')
# print(f' - Etiquetas: {Y_train.shape}')

# print('Tamaño de los datos de test:')
# print(f' - Imágenes: {X_test.shape}')
# print(f' - Etiquetas: {Y_test.shape}')

# # Muestra la imagen de ejemplo
# id_ejemplo = 10
# plt.imshow(X_train[id_ejemplo], cmap='gray')
# plt.show()
# print(f'Ejemplo de etiqueta: {Y_train[id_ejemplo]}')

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train /= 255.
X_test /= 255.

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Conversión a one-hot
Y_train = tf.keras.utils.to_categorical(Y_train) 
Y_test = tf.keras.utils.to_categorical(Y_test)

print('Tamaño de los datos de entrenamiento:')
print(f' - Imágenes: {X_train.shape}')
print(f' - Etiquetas: {Y_train.shape}')

print('Tamaño de los datos de test:')
print(f' - Imágenes: {X_test.shape}')
print(f' - Etiquetas: {Y_test.shape}')

# Muestra la imagen de ejemplo
# id_ejemplo = 10
# print(f'Ejemplo de etiqueta: {Y_train[id_ejemplo]}')

# Red neuronal
# modelo = tf.keras.Sequential()
# modelo.add(tf.keras.layers.Input(shape=(784,)))
# modelo.add(tf.keras.layers.Dense(512, activation='relu'))
# modelo.add(tf.keras.layers.Dense(10, activation='softmax'))

# print(modelo.summary())

# modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# historial = modelo.fit(X_train, Y_train, validation_split=0.25, epochs=15, batch_size=32, verbose=1)

# print(historial.history.keys())

# plt.plot(historial.history['accuracy'])
# plt.plot(historial.history['val_accuracy'])
# plt.xlabel('Época')
# plt.ylabel('Tasa de acierto')
# plt.legend(['Entrenamiento', 'Validación'])
# plt.show()

# print(np.argmax(modelo.predict(X_test), axis=-1))
# loss, accuracy = modelo.evaluate(X_test, Y_test, verbose=0)
# print('Tasa de acierto sobre test: {:.2f}%'.format(100*accuracy))

# modelo.save('modelo_cap_4.h5')

# Cargar red neuronal de un archivo .h5
nuevo_modelo = tf.keras.models.load_model('modelo_cap_4.h5')
print(nuevo_modelo.summary())

loss, accuracy = nuevo_modelo.evaluate(X_test, Y_test, verbose=0)
print('Tasa de acierto sobre test: {:.2f}%'.format(100*accuracy))

