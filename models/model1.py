import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from models.DataGenerator import DataGenerator as dg

# define the keras model
def run_model(X, y, dim_embeddings, epochs, batch_size):
    model = keras.Sequential()
    input_users = keras.layers.Input(shape=(dim_embeddings,))
    x1 = keras.layers.Dense(512, activation=tf.nn.relu)(input_users)
    x1_2 = keras.layers.Dense(256, activation=tf.nn.relu)(x1)
    x1_3 = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2)

    input_items = keras.layers.Input(shape=(dim_embeddings,))
    x2 = keras.layers.Dense(512, activation=tf.nn.relu)(input_items)
    x2_2 = keras.layers.Dense(256, activation=tf.nn.relu)(x2)
    x2_3 = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2)

    concatenated = keras.layers.Concatenate()([x1_3, x2_3])
    d1 = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated)
    d2 = keras.layers.Dense(64, activation=tf.nn.relu)(d1)
    out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(d2)
    model = keras.models.Model(inputs=[input_users,input_items],outputs=out)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
    model.fit([X[:,0],X[:,1]], y, epochs=epochs, batch_size=batch_size)


    return model