from utilities.utils import read_ratings,read_graph_embeddings,read_bert_embedding,matching_userBert_itemGraph,matching_userGraph_itemBert,top_scores
from models.model1 import run_model
import logging
import sys

if __name__ == "__main__":
  logging.basicConfig(format="%(message)s", level=logging.INFO)
  logger = logging.getLogger(__name__)
  if len(sys.argv) != 6:
    logger.error("Invalid number of parameters.")
    exit(-1)
  
user_source = sys.argv[1]
item_source = sys.argv[2]
dest = sys.argv[3]
prediction_dest = sys.argv[4]
isUserGraph = int(sys.argv[5])

print(user_source)
print(item_source)
print(dest)
print(prediction_dest)

user, item, rating = read_ratings('datasets/movielens/train2id.tsv')

if isUserGraph == 1:	
	print("User is encoded with graph embedding")
	user_embeddings = read_graph_embeddings(user_source)
	item_embeddings = read_bert_embedding(item_source)
	X, y, dim_embeddings = matching_userGraph_itemBert(user,item,rating,user_embeddings,item_embeddings)
else:
	print("User is encoded with bert embedding")
	item_embeddings = read_graph_embeddings(item_source)
	user_embeddings = read_bert_embedding(user_source) 
	X, y, dim_embeddings = matching_userBert_itemGraph(user,item,rating,user_embeddings,item_embeddings)

model = run_model(X, y, dim_embeddings, epochs=25, batch_size=512)

# creates a HDF5 file 'model.h5'
model.save(dest + 'model.h5')
