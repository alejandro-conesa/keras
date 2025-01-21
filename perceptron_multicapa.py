import tensorflow as tf

# Ejercicio de ejemplo sobre diferentes funciones de activación y el uso de múltiples capas

# Creación y definición inicial de la red neuronal
capa_entrada = tf.keras.layers.Input(shape=(10))
capa_oculta1 = tf.keras.layers.Dense(units=20, activation='relu')(capa_entrada)
capa_oculta2 = tf.keras.layers.Dense(units=30, activation='tanh')(capa_oculta1)
capa_salida = tf.keras.layers.Dense(units=1, activation='sigmoid')(capa_oculta2)

perceptron_multicapa = tf.keras.Model(inputs = capa_entrada, outputs = capa_salida)