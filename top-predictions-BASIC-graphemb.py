from utilities.utils import read_graph_embeddings, read_ratings, matching_graph_emb_id, top_scores
from models.model1 import run_model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import sys

if __name__ == "__main__":
  logging.basicConfig(format="%(message)s", level=logging.INFO)
  logger = logging.getLogger(__name__)
  if len(sys.argv) != 4:
    logger.error("Invalid number of parameters.")
    exit(-1)
  
source = sys.argv[1]
dest = sys.argv[2]
prediction_dest = sys.argv[3]


model = tf.keras.models.load_model(dest + 'model.h5')

ent_embeddings = read_graph_embeddings(source)
user, item, rating = read_ratings('datasets/movielens/test2id.tsv')
X, y, dim_embeddings = matching_graph_emb_id(user, item, rating, ent_embeddings)

score = model.predict([X[:,0],X[:,1]])

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