import pandas as pd
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#normalized_method = "Laplacian"
normalized_method = "endocrine"

exp_pattern = "exp2a"
#exp_pattern = "exp2b"
#exp_pattern = "exp3"


"""
各予測パターンに含まれる遺伝子ペアの発現をプロットする
"""
df_heat = pd.read_table("result/consideration/heatmap/"+normalized_method+"/"+exp_pattern+"_positive.csv", sep=',', index_col=0)
spatial = pd.read_csv("data/visium_data/tissue_positions_list.csv", header=None)
clusters = pd.read_table("data/cellphonedb_input/clusters_reshaped.txt")
counts = pd.read_table("data/cellphonedb_input/counts_reshaped.txt", index_col="GENE").T

all_spot_name = spatial.iloc[:,0].tolist()
spot_name = clusters.iloc[:,0].tolist()
number = []
for i in range(len(spot_name)):
  number.append(all_spot_name.index(spot_name[i]))
df_spatial=spatial.iloc[number, [0,4,5]].set_axis(['name','x','y'], axis=1)


for pattern in ["1111","1110","1101","1011","0111","1100","1010","1001","0110","0101","0011","1000","0100","0010","0001","0000"]:
  # 各予測パターンの遺伝子ペアをリストに保存
  pair_list=[]
  for i in range(len(df_heat)):
    if str(int(df_heat.iloc[i]["sum"])).zfill(4)==pattern:
      pair_list.append(df_heat.index[i])
  
  if len(pair_list)>0:
    save_path = 'result/consideration/expression_image/'+normalized_method+"_"+exp_pattern+'/'+pattern+"("+str(len(pair_list))+")/"
    if not os.path.isdir(save_path):
      os.makedirs(save_path)

  # 各遺伝ペアの発現をプロット
  for index in range(len(pair_list)):
    genes_list = pair_list[index].split('-')

    G1 = nx.Graph()
    G2 = nx.Graph()

    ## ノードのラベルを定義
    node_labels = []
    for i in range(len(df_spatial.loc[:,["x","y"]])):
      node_labels.append(str(i))
    G1.add_nodes_from(node_labels)
    G2.add_nodes_from(node_labels)

    ## 各ノードの座標を定義
    x_list=df_spatial.loc[:,"x"].tolist()
    y_list=df_spatial.loc[:,"y"].tolist()
    pos = {
      n: (x_list[i], y_list[i])
      for i, n in enumerate(G1.nodes)
    }

    nodesize1 = counts[genes_list[0]]
    nodesize2 = counts[genes_list[1]]
    
    li = []
    li.extend(nodesize1)
    li.extend(nodesize2)
    li = np.array(li)

    li = preprocessing.minmax_scale(li)
    li = map(lambda x: x * 3500, li)
    li = list(li)
    
    plt.figure(figsize=(55,55))
    nx.draw_networkx(G1, pos=pos, node_size=li[0:len(nodesize1)], node_color="r", with_labels=False, alpha=0.5)
    nx.draw_networkx(G2, pos=pos, node_size=li[len(nodesize1):len(li)], node_color="b", with_labels=False, alpha=0.5)
    plt.savefig(save_path+pair_list[index]+".pdf")
    plt.close()



