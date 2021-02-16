from utilities.utils import read_bert_embeddings,read_ratings,matching_bert_emb_id
from models.model1 import run_model
import pandas as pd
import numpy as np
import logging
import sys

if __name__ == "__main__":
  logging.basicConfig(format="%(message)s", level=logging.INFO)
  logger = logging.getLogger(__name__)
  if len(sys.argv) != 5:
    logger.error("Invalid number of parameters.")
    exit(-1)
  
user_source = sys.argv[1]
item_source = sys.argv[2]
dest = sys.argv[3]
prediction_dest = sys.argv[4]

print(user_source)
print(item_source)
print(dest)
print(prediction_dest)
  
user_embeddings, item_embeddings = read_bert_embeddings(user_source, item_source)
user, item, rating = read_ratings('datasets/movielens/train2id.tsv')
X, y, dim_embeddings = matching_bert_emb_id(user,item,rating,user_embeddings,item_embeddings)

model = run_model(X, y, dim_embeddings, epochs=25, batch_size=512)

# creates a HDF5 file 'model.h5'
model.save(dest + 'model.h5')
