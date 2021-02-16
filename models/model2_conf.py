import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#CONFIGURAZIONE 1
def run_conf_1(X,y,dim_embeddings,epochs,batch_size):
  model = keras.Sequential()
  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))
  x1_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_1)
  x1_2_user = keras.layers.Dense(64, activation=tf.nn.relu)(x1_user)


  x1_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_1)
  x1_2_item = keras.layers.Dense(64, activation=tf.nn.relu)(x1_item)


  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))
  x2_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_2)
  x2_2_user = keras.layers.Dense(64, activation=tf.nn.relu)(x2_user)

  x2_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_2)
  x2_2_item = keras.layers.Dense(64, activation=tf.nn.relu)(x2_item)
  
  concatenated_1 = keras.layers.Concatenate()([x1_2_user, x2_2_user])
  dense_user = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)
  dense_user_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_user)
  
  concatenated_2 = keras.layers.Concatenate()([x1_2_item, x2_2_item])
  dense_item = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)
  dense_item_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_item)

  concatenated = keras.layers.Concatenate()([dense_user_2, dense_item_2])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  dense2 = keras.layers.Dense(8, activation=tf.nn.relu)(dense)


  #concatenated = keras.layers.Flatten()(concatenated)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense2)
  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  
  return model

#CONFIGURAZIONE 2
def run_conf_2(X,y,dim_embeddings,epochs,batch_size):
  model = keras.Sequential()
  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))
  x1_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_1)
  x1_2_user = keras.layers.Dense(64, activation=tf.nn.relu)(x1_user)


  x1_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_1)
  x1_2_item = keras.layers.Dense(64, activation=tf.nn.relu)(x1_item)


  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))
  x2_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_2)
  x2_2_user = keras.layers.Dense(64, activation=tf.nn.relu)(x2_user)

  x2_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_2)
  x2_2_item = keras.layers.Dense(64, activation=tf.nn.relu)(x2_item)
  
  concatenated_1 = keras.layers.Concatenate()([x1_2_user, x1_2_item])
  dense_user = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)
  dense_user_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_user)
  
  concatenated_2 = keras.layers.Concatenate()([x2_2_user, x2_2_item])
  dense_item = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)
  dense_item_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_item)

  concatenated = keras.layers.Concatenate()([dense_user_2, dense_item_2])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  dense2 = keras.layers.Dense(8, activation=tf.nn.relu)(dense)


  #concatenated = keras.layers.Flatten()(concatenated)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense2)
  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  
  return model
