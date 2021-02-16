import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#CONFIGURAZIONE 2
def run_model(X,y,dim_embeddings,epochs,batch_size):
  model = keras.Sequential()
  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))
  x1_user = keras.layers.Dense(50, activation=tf.nn.relu)(input_users_1)

  attention_probs_u1 = keras.layers.Dense(50, activation='softmax')(x1_user)
  attention_mul_u1 = keras.layers.multiply([x1_user, attention_probs_u1])


  x1_item = keras.layers.Dense(50, activation=tf.nn.relu)(input_items_1)

  attention_probs_i1 = keras.layers.Dense(50, activation='softmax')(x1_item)
  attention_mul_i1 = keras.layers.multiply([x1_item, attention_probs_i1])

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))
  x2_user = keras.layers.Dense(50, activation=tf.nn.relu)(input_users_2)
  attention_probs_u2 = keras.layers.Dense(50, activation='softmax')(x2_user)
  attention_mul_u2 = keras.layers.multiply([x2_user, attention_probs_u2])

  x2_item = keras.layers.Dense(50, activation=tf.nn.relu)(input_items_2)
  attention_probs_i2 = keras.layers.Dense(50, activation='softmax')(x2_item)
  attention_mul_i2 = keras.layers.multiply([x2_item, attention_probs_i2])
  #user graph + item graph
  concatenated_1 = keras.layers.Concatenate()([attention_mul_u1, attention_mul_i1])

  #user bert + item bert
  concatenated_2 = keras.layers.Concatenate()([attention_mul_u2, attention_mul_i2])

  concatenated = keras.layers.Concatenate()([concatenated_1, concatenated_2])


  '''
  #BLOCCO DI ATTENZIONE 2
  attention_probs = keras.layers.Dense(200, activation='softmax')(attention_mul)
  attention_mul = keras.layers.multiply([attention_mul, attention_probs])

  #BLOCCO DI ATTENZIONE 3
  attention_probs = keras.layers.Dense(200, activation='softmax')(attention_mul)
  attention_mul = keras.layers.multiply([attention_mul, attention_probs])
  '''

  dense_layer = keras.layers.Dense(100, activation=tf.nn.relu)(concatenated)

  '''
  dropout = keras.layers.Dropout(0.2)(dense_layer)
  dense_layer = keras.layers.Dense(100, activation=tf.nn.relu)(dropout)
  '''

  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense_layer)
  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  print(model.summary())
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  return model
