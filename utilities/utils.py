import pandas as pd
import csv
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def read_bert_embeddings(filename_users,filename_items):
  #reading embeddings
  user_embeddings =  pd.read_json(filename_users)
  item_embeddings =  pd.read_json(filename_items)
  user_embeddings = user_embeddings.sort_values(by=["ID_OpenKE"])
  item_embeddings = item_embeddings.sort_values(by=["ID_OpenKE"])
  return user_embeddings,item_embeddings

def read_bert_embedding(filename_bert):
  #reading embeddings
  bert_embeddings =  pd.read_json(filename_bert)
  bert_embeddings = bert_embeddings.sort_values(by=["ID_OpenKE"])
  return bert_embeddings

def read_graph_embeddings(filename_graph):
  #reading embeddings
  with open(filename_graph) as json_file:
    embeddings = json.load(json_file)
    ent_embeddings = embeddings['ent_embeddings']
  return ent_embeddings

def read_ratings(filename):
  user=[]
  item=[]
  rating=[]
  #reading item ids
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    #next(csv_reader)
    for row in csv_reader:
        user.append(int(row[0]))
        item.append(int(row[1]))
        rating.append(int(row[2]))
  return user, item, rating

def matching_graph_emb_id(user,item,rating,ent_embeddings):
  y = np.array(rating)
  dim_embeddings = len(ent_embeddings[0])
  dim_X_cols = 2
  dim_X_rows = len(user)
  X = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings))
  #matching between ids and embeddings
  i=0
  while i < dim_X_rows:
      user_id = user[i]
      item_id = item[i]
      X[i][0] = ent_embeddings[user_id]
      X[i][1] = ent_embeddings[item_id]
      i=i+1
  return X, y, dim_embeddings

def matching_bert_emb_id(user,item,rating,user_embeddings,item_embeddings):
  y = np.array(rating)
  dim_embeddings = len(item_embeddings['embedding'][0])
  dim_X_cols = 2
  dim_X_rows = len(user)
  X = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings))
  #matching between ids and embeddings
  i=0
  while i < dim_X_rows:
      user_id = user[i]
      item_id = item[i]
      #print(str(user_id)+"_"+str(item_id))
      user_embedding = (np.array(user_embeddings.loc[(user_embeddings["ID_OpenKE"]==user_id),"profile_embedding"]))[0]
      item_embedding = (np.array(item_embeddings.loc[(item_embeddings["ID_OpenKE"]==item_id),"embedding"]))[0]
      X[i][0] = user_embedding
      X[i][1] = item_embedding
      i=i+1

  return X, y, dim_embeddings

def matching_userBert_itemGraph(user,item,rating,user_embeddings,item_embeddings):
  y = np.array(rating)
  dim_embeddings = len(user_embeddings['profile_embedding'][0])
  dim_X_cols = 2
  dim_X_rows = len(user)
  X = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings))
  #matching between ids and embeddings
  i=0
  while i < dim_X_rows:
      user_id = user[i]
      item_id = item[i]
      user_embedding = (np.array(user_embeddings.loc[(user_embeddings["ID_OpenKE"]==user_id),"profile_embedding"]))[0]
      X[i][0] = user_embedding
      X[i][1] = item_embeddings[item_id]
      i=i+1

  return X, y, dim_embeddings

def matching_userGraph_itemBert(user,item,rating,user_embeddings,item_embeddings):
  y = np.array(rating)
  dim_embeddings = len(item_embeddings['embedding'][0])
  dim_X_cols = 2
  dim_X_rows = len(user)
  X = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings))
  #matching between ids and embeddings
  i=0
  while i < dim_X_rows:
      user_id = user[i]
      item_id = item[i]
      item_embedding = (np.array(item_embeddings.loc[(item_embeddings["ID_OpenKE"]==item_id),"embedding"]))[0]
      X[i][0] = user_embeddings[user_id]
      X[i][1] = item_embedding
      i=i+1

  return X, y, dim_embeddings

def matching_Bert_Graph_conf(user,item,rating,graph_embeddings,user_bert_embeddings,item_bert_embeddings):
  y = np.array(rating)
  dim_embeddings = len(item_bert_embeddings['embedding'][0])
  dim_X_cols = 4
  dim_X_rows = len(user)
  X = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings))
  #matching between ids and embeddings (graph and bert)
  i=0
  while i < dim_X_rows:
      user_id = user[i]
      item_id = item[i]
      user_bert_embedding = (np.array(user_bert_embeddings.loc[(user_bert_embeddings["ID_OpenKE"]==user_id),"profile_embedding"]))[0]
      item_bert_embedding = (np.array(item_bert_embeddings.loc[(item_bert_embeddings["ID_OpenKE"]==item_id),"embedding"]))[0]
      X[i][0] = graph_embeddings[user_id] #user graph
      X[i][1] = graph_embeddings[item_id] #item graph
      X[i][2] = user_bert_embedding #user bert
      X[i][3] = item_bert_embedding #item bert
      i=i+1

  return X, y, dim_embeddings

def matching_Bert_Graph(user,item,rating,graph_embeddings,user_bert_embeddings,item_bert_embeddings):
  y = np.array(rating)
  dim_embeddings_bert = len(item_bert_embeddings['embedding'][0])
  dim_embeddings_graph = len(graph_embeddings[0])
  dim_X_cols = 2
  dim_X_rows = len(user)
  X_graph = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings_graph))
  X_bert = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings_bert))
  #matching between ids and embeddings (graph and bert)
  i=0
  while i < dim_X_rows:
      user_id = user[i]
      item_id = item[i]
      user_bert_embedding = (np.array(user_bert_embeddings.loc[(user_bert_embeddings["ID_OpenKE"]==user_id),"profile_embedding"]))[0]
      item_bert_embedding = (np.array(item_bert_embeddings.loc[(item_bert_embeddings["ID_OpenKE"]==item_id),"embedding"]))[0]
      X_graph[i][0] = graph_embeddings[user_id] #user graph
      X_graph[i][1] = graph_embeddings[item_id] #item graph
      X_bert[i][0] = user_bert_embedding #user bert
      X_bert[i][1] = item_bert_embedding #item bert
      i=i+1

  return X_graph, X_bert, dim_embeddings_graph, dim_embeddings_bert, y

def top_scores(predictions,n):
  top_n_scores = pd.DataFrame()
  for u in list(set(predictions['users'])):
    p = predictions.loc[predictions['users'] == u ]
    top_n_scores = top_n_scores.append(p.head(n))
  return top_n_scores

