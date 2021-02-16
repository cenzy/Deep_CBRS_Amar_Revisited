import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#CONFIGURAZIONE 2 - strategia 1
def run_model(X_graph,X_bert,dim_graph,dim_bert,y,epochs,batch_size):
  model = keras.Sequential()
  #input_users_1 = keras.layers.Input(shape=(dim_graph,))
  input_items_1 = keras.layers.Input(shape=(dim_graph,))
  #x1_user = input_users_1
  x1_item = input_items_1

  input_users_2 = keras.layers.Input(shape=(dim_bert,))
  #input_items_2 = keras.layers.Input(shape=(dim_bert,))
  x2_user = keras.layers.Dense(768, activation=tf.nn.relu)(input_users_2)
  #x2_item = keras.layers.Dense(768, activation=tf.nn.relu)(input_items_2)

  #user graph + item bert
  #concatenated_1 = keras.layers.Concatenate()([x1_user, x2_item])

  #user bert + item graph
  concatenated_2 = keras.layers.Concatenate()([x2_user, x1_item])

  #dense after concat
  #dense = keras.layers.Dense(8, activation=tf.nn.relu)(concatenated_1)
  dense = keras.layers.Dense(8, activation=tf.nn.relu)(concatenated_2)

  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense)
  model = keras.models.Model(inputs=[input_users_2,input_items_1],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X_bert[:,0],X_graph[:,1]], y, epochs=epochs, batch_size=batch_size)
  return model