import tensorflow as tf

# Ejercicio de ejemplo sobre diferentes tipos de clasificación

# Creación y definición inicial de la red neuronal
capa_entrada = tf.keras.layers.Input(shape=(10,))
capa_oculta1 = tf.keras.layers.Dense(units=20, activation='relu')(capa_entrada)
capa_oculta2 = tf.keras.layers.Dense(units=30, activation='tanh')(capa_oculta1)

# Modelo 1 neurona
binaria_1neurona = tf.keras.layers.Dense(units=1, activation='sigmoid')(capa_oculta2)
modelo_1neurona = tf.keras.Model(inputs=capa_entrada, outputs=binaria_1neurona)

# Modelo 2 neuronas
binaria_2neuronas = tf.keras.layers.Dense(units=2, activation='softmax')(capa_oculta2)
modelo_2neuronas = tf.keras.Model(inputs=capa_entrada, outputs=binaria_2neuronas)

# Modelo multiclase (generalización de 2 neuronas)
multiclase = tf.keras.layers.Dense(units=5, activation='softmax')(capa_oculta2)
modelo_multiclase = tf.keras.Model(inputs=capa_entrada, outputs=multiclase)

# Modelo multietiqueta
multietiqueta = tf.keras.layers.Dense(units=5, activation='softmax')(capa_oculta2)
modelo_multietiqueta = tf.keras.Model(inputs=capa_entrada, outputs=multietiqueta)

# Ejecución de ejemplo
tensor = tf.random.normal(shape=(1,10))

salida_1neurona = modelo_1neurona(tensor)
salida_2neuronas = modelo_2neuronas(tensor)
salida_multiclase = modelo_multiclase(tensor)
salida_multietiqueta = modelo_multietiqueta(tensor)

print(f'Binaria una neurona - Clase {"0" if salida_1neurona > 0.5 else "1"}')
print(f'Binaria dos neurona - Clase {tf.argmax(salida_2neuronas, axis =-1)}')
print(f'Multiclase - Clase {tf.argmax(salida_multiclase, axis =-1)}')
print(f'Multietiqueta - Clase {tf.where(salida_multietiqueta[0] > 0.5)[:, 0]}')

