import tensorflow as tf

capa_entrada = tf.keras.layers.Input(shape=(10))

capa_oculta1 = tf.keras.layers.Dense(units=20, activation='relu')(capa_entrada)
capa_oculta2 = tf.keras.layers.Dense(units=30, activation='tanh')(capa_oculta1)

capa_salida = tf.keras.layers.Dense(units=1, activation='sigmoid')(capa_oculta2)