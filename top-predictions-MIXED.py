import pandas as pd
import csv
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from utilities.utils import read_ratings,read_graph_embeddings,read_bert_embedding,top_scores,matching_Bert_Graph_conf
import logging
import sys

if __name__ == "__main__":
  logging.basicConfig(format="%(message)s", level=logging.INFO)
  logger = logging.getLogger(__name__)
  if len(sys.argv) != 6:
    logger.error("Invalid number of parameters.")
    exit(-1)
  
bert_user_source = sys.argv[1]
bert_item_source = sys.argv[2]
graph_source = sys.argv[3]
dest = sys.argv[4]
prediction_dest = sys.argv[5]

print(bert_user_source)
print(bert_item_source)
print(graph_source)
print(dest)
print(prediction_dest)

user, item, rating = read_ratings('datasets/movielens/test2id.tsv')
model = tf.keras.models.load_model(dest + 'model.h5')

graph_embeddings = read_graph_embeddings(graph_source)
user_bert_embeddings = read_bert_embedding(bert_user_source)
item_bert_embeddings = read_bert_embedding(bert_item_source)

X, y, dim_embeddings = matching_Bert_Graph_conf(user,item,rating,graph_embeddings,user_bert_embeddings,item_bert_embeddings)

score = model.predict([X[:,0],X[:,1],X[:,2],X[:,3]])

print("Computing predictions...")
score = score.reshape(1, -1)[0,:]
predictions = pd.DataFrame()
predictions['users'] = np.array(user) + 1
predictions['items'] = np.array(item) + 1
predictions['scores'] = score

predictions = predictions.sort_values(by=['users', 'scores'],ascending=[True, False])

top_5_scores = top_scores(predictions,5)
top_5_scores.to_csv(prediction_dest + 'top_5/predictions_1.tsv',sep='\t',header=False,index=False)
print("Successfully writing top 5 scores")

top_10_scores = top_scores(predictions,10)
top_10_scores.to_csv(prediction_dest + 'top_10/predictions_1.tsv',sep='\t',header=False,index=False)
print("Successfully writing top 10 scores")