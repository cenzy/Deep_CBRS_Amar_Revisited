import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
from utilities.utils import read_ratings,read_graph_embeddings,read_bert_embedding,matching_Bert_Graph_conf
from models.model3_conf2_att import run_model
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

user, item, rating = read_ratings('datasets/movielens/train2id.tsv')

graph_embeddings = read_graph_embeddings(graph_source)
user_bert_embeddings = read_bert_embedding(bert_user_source)
item_bert_embeddings = read_bert_embedding(bert_item_source)

X, y, dim_embeddings = matching_Bert_Graph_conf(user,item,rating,graph_embeddings,user_bert_embeddings,item_bert_embeddings)

model = run_model(X,y,dim_embeddings,epochs=30,batch_size=512)

# creates a HDF5 file 'model.h5'
model.save(dest + 'model.h5')