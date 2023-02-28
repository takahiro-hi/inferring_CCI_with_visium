import pandas as pd
import numpy as np
import scipy.spatial.distance as distance
from scipy.spatial.distance import pdist
import methods_processing_matrix as methods
import os
import pickle


spatial = pd.read_csv("data/visium_data/tissue_positions_list.csv", header=None)
clusters = pd.read_table("data/cellphonedb_input/clusters_reshaped.txt")

x = spatial.loc[:,4].tolist()
y = spatial.loc[:,5].tolist()
all_spot_name = spatial.iloc[:,0].tolist()
spot_name = clusters.loc[:,'Barcode'].tolist()
cluster_list = clusters.loc[:,'Cluster'].tolist()

"""
発現領域のスポット名・座標をdf_spatialに保存
"""
index_target = []
for i in range(len(spot_name)):
  index_target.append(all_spot_name.index(spot_name[i]))
df_spatial=spatial.iloc[index_target, [0,4,5]].set_axis(['name','x','y'], axis=1)


"""
距離行列を作成し，隣接行列を4パターン作成
"""
distance_matrix = distance.squareform(distance.pdist(df_spatial.loc[:,["x","y"]]))
adj1 = np.zeros((int(distance_matrix.shape[0]), int(distance_matrix.shape[1])))
adj2 = np.zeros((int(distance_matrix.shape[0]), int(distance_matrix.shape[1])))
adj3 = np.zeros((int(distance_matrix.shape[0]), int(distance_matrix.shape[1])))
adj4 = np.zeros((int(distance_matrix.shape[0]), int(distance_matrix.shape[1])))

for i in range(distance_matrix.shape[0]):
  for j in range(distance_matrix.shape[1]):
    if i==j:
      adj1[i,j] = 1
    if 0<distance_matrix[i,j] and distance_matrix[i,j]<300:
      adj2[i,j] = 1
      if cluster_list[i]==cluster_list[j]:
        adj4[i,j] = 1
    if 0<distance_matrix[i,j] and distance_matrix[i,j]<600:
      adj3[i,j] = 1


"""
グラフを可視化
"""
methods.make_adj_graph(adj1, df_spatial, "adj1")
methods.make_adj_graph(adj2, df_spatial, "adj2")
methods.make_adj_graph(adj3, df_spatial, "adj3")
methods.make_adj_graph(adj4, df_spatial, "adj4")


"""
隣接行列を正規化して保存
"""
save_path = 'data/gcng_input/adj_matrix/adj1/'
if not os.path.isdir(save_path):
  os.makedirs(save_path)

with open(save_path+'endocrine', 'wb') as fp:
  pickle.dump(adj1, fp)
methods.normalized_adj_matrix_endocrine(adj2, "adj2")
methods.normalized_adj_matrix_endocrine(adj3, "adj3")
methods.normalized_adj_matrix_endocrine(adj4, "adj4")

with open(save_path+'autocrine', 'wb') as fp:
  pickle.dump(adj1, fp)
methods.normalized_adj_matrix_autocrine(adj2, "adj2")
methods.normalized_adj_matrix_autocrine(adj3, "adj3")
methods.normalized_adj_matrix_autocrine(adj4, "adj4")

with open(save_path+'Laplacian', 'wb') as fp:
  pickle.dump(adj1, fp)
methods.normalized_adj_matrix_Laplacian(adj2, "adj2")
methods.normalized_adj_matrix_Laplacian(adj3, "adj3")
methods.normalized_adj_matrix_Laplacian(adj4, "adj4")

with open(save_path+'normal', 'wb') as fp:
  pickle.dump(adj1, fp)
methods.normalized_adj_matrix(adj2, "adj2")
methods.normalized_adj_matrix(adj3, "adj3")
methods.normalized_adj_matrix(adj4, "adj4")
