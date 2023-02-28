import networkx as nx
import matplotlib.pyplot as plt
import os
import spektral
import numpy as np
from scipy import sparse
import pickle


"""
隣接行列とスポット座標を引数により，グラフをプロットする関数
"""
def make_adj_graph(adj, df_spatial, name):
  save_path = 'data/gcng_input/adj_matrix/image/'
  if not os.path.isdir(save_path):
    os.makedirs(save_path)

  G = nx.Graph()

  # ノードのラベルを定義
  node_labels = []
  for i in range(len(df_spatial)):
    node_labels.append(str(i))
  G.add_nodes_from(node_labels)

  # スポット間のエッジを定義
  for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
      if adj[i,j]==1:
        G.add_edge(node_labels[i],node_labels[j])

  # 各ノードの座標を定義
  x_list=df_spatial.loc[:,"x"].tolist()
  y_list=df_spatial.loc[:,"y"].tolist()
  pos = {
    n: (x_list[i], y_list[i])
    for i, n in enumerate(G.nodes)
  }

  # グラフを描写
  fig = plt.figure(figsize=(55,55))
  nx.draw_networkx(G, pos=pos, node_color="b")
  fig.savefig(save_path+name+".pdf")
  plt.clf()



"""
隣接行列を正規化 (Endocrine GCNGのパターン)
  A_N  = D^(-1/2) A D^(-1/2)
  L_NN = I-D_N^(-1/2) A_N D_N^(-1/2)
"""
def normalized_adj_matrix_endocrine(adj, name):
  adj_I_N = spektral.utils.normalized_adjacency(adj, symmetric=True)
  adj_I_N = spektral.utils.normalized_laplacian(adj_I_N, symmetric=True)
  adj_I_N = np.float32(adj_I_N)
  adj_I_N_crs = sparse.csr_matrix(adj_I_N)
  
  save_path = 'data/gcng_input/adj_matrix/'+name+'/'
  if not os.path.isdir(save_path):
    os.makedirs(save_path)
  with open(save_path+'endocrine', 'wb') as fp:
    pickle.dump(adj_I_N_crs, fp)


"""
隣接行列を正規化 (Autocrine+ GCNGのパターン)
  A' = A + I
  L' = D'^(-1/2) A' D'^(-1/2)
"""
def normalized_adj_matrix_autocrine(adj, name):
  I = sparse.eye(adj.shape[-1], dtype=adj.dtype)
  adj_I   = adj + I
  adj_I_N = spektral.utils.normalized_adjacency(adj_I, symmetric=True)
  adj_I_N = np.float32(adj_I_N)
  adj_I_N_crs = sparse.csr_matrix(adj_I_N)
    
  save_path = 'data/gcng_input/adj_matrix/'+name+'/'
  if not os.path.isdir(save_path):
    os.makedirs(save_path)
  with open(save_path+'autocrine', 'wb') as fp:
    pickle.dump(adj_I_N_crs, fp)


"""
隣接行列を正規化 (正規化グラフラプラシアン)
  L = I - D^(-1/2) A D^(-1/2)
"""
def normalized_adj_matrix_Laplacian(adj, name):
  adj_I_N = spektral.utils.normalized_laplacian(adj, symmetric=True)
  adj_I_N = np.float32(adj_I_N)
  adj_I_N_crs = sparse.csr_matrix(adj_I_N)

  save_path = 'data/gcng_input/adj_matrix/'+name+'/'
  if not os.path.isdir(save_path):
    os.makedirs(save_path)
  with open(save_path+'Laplacian', 'wb') as fp:
    pickle.dump(adj_I_N_crs, fp)


"""
隣接行列を正規化 (正規化)
  A_N  = D^(-1/2) A D^(-1/2)
"""
def normalized_adj_matrix(adj, name):
  adj_I_N = spektral.utils.normalized_adjacency(adj, symmetric=True)
  adj_I_N = np.float32(adj_I_N)
  adj_I_N_crs = sparse.csr_matrix(adj_I_N)

  save_path = 'data/gcng_input/adj_matrix/'+name+'/'
  if not os.path.isdir(save_path):
    os.makedirs(save_path)
  with open(save_path+'normal', 'wb') as fp:
    pickle.dump(adj_I_N_crs, fp)





