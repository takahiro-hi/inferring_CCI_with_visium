import pandas as pd
import collections
import matplotlib.pyplot as plt
import os
import networkx as nx
import random
import numpy as np

pvalue=pd.read_table("data/cellphonedb_output/pvalues.txt")
counts = pd.read_table("data/cellphonedb_input/counts_reshaped.txt", index_col="GENE").T

"""
index_not_nan : gene_aとgene_bが共にNANではないペアのインデント
"""
index_not_nan = []
for i in range(len(pvalue)):
  if type(pvalue.iloc[i]["gene_a"]) is str:
    if type(pvalue.iloc[i]["gene_b"]) is str:
      index_not_nan.append(i)

"""
all_pair : 重複を除いた遺伝子ペア全て
"""
all_pair = []
for i in index_not_nan:
  all_pair.append((pvalue.iloc[int(i),4], pvalue.iloc[int(i),5]))

for i in range(len(all_pair)):
  sorted(all_pair[i])

all_pair = list(set(all_pair))

"""
all_gene : 全ての遺伝子
"""
all_gene=[]
for i in range(len(all_pair)):
  all_gene.append(all_pair[i][0])
  all_gene.append(all_pair[i][1])
all_gene = list(set(all_gene))


"""
評価用データセット1, 2a, 2b, 3 を作成
"""
for pattern in [1,2,3]:
  # positiveリストをダウンロード
  with open("data/gcng_input/interaction_list/positive_"+str(pattern)+".txt") as f:
    index_positive = f.read().split(', ')
  index_positive[0] = index_positive[0].lstrip("[")
  index_positive[len(index_positive)-1] = index_positive[len(index_positive)-1].rstrip("]")

  # positiveリストを遺伝子ペアの形で(ソートして) positive_pairs に格納
  positive_pairs = []
  for i in index_positive:
    positive_pairs.append((pvalue.loc[int(i),'gene_a'], pvalue.loc[int(i),'gene_b']))
  for i in range(len(positive_pairs)):
    sorted(positive_pairs[i])

  # negativeリストを遺伝子ペアの形で(ソートして) negative_pairs に格納
  negative_pairs = []
  random.seed(1)
  for i in range(len(positive_pairs)):
    while(1):
      pair = tuple(sorted(random.choices(all_gene,k=2)))
      if not pair in all_pair:
        negative_pairs.append(pair)
        break
  
  if pattern!=2:
    print("[dataset "+str(pattern)+"(0~"+str(len(positive_pairs))+")]")
  else:
    print("[dataset "+str(pattern)+"a(0~"+str(len(positive_pairs))+")]")

  positive_pairs = random.sample(positive_pairs,len(positive_pairs))
  negative_pairs = random.sample(negative_pairs,len(negative_pairs))
  
  # 評価用データセット1, 2a, 3を作成
  for i in range(1,4):
    index = [i for i in range(len(positive_pairs))]
    test_index = [i for i in range(int((i-1)*len(index)/3), int(i*len(index)/3))]
    train_index  = [i for i in index if i not in test_index]

    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    train_pair_list=[]
    test_pair_list=[]

    for j in train_index:
      X_train.append(np.array(counts[[positive_pairs[j][0], positive_pairs[j][1]]]))
      X_train.append(np.array(counts[[negative_pairs[j][0], negative_pairs[j][1]]]))
      y_train.append(1)
      y_train.append(0)
      train_pair_list.append(positive_pairs[j])
      train_pair_list.append(negative_pairs[j])
  
    for j in test_index:
      X_test.append(np.array(counts[[positive_pairs[j][0], positive_pairs[j][1]]]))
      X_test.append(np.array(counts[[negative_pairs[j][0], negative_pairs[j][1]]]))
      y_test.append(1)
      y_test.append(0)
      test_pair_list.append(positive_pairs[j])
      test_pair_list.append(negative_pairs[j])

    print("  <"+str(i)+"> train:", len(X_train), ", test:", len(X_test), " ("+str(int((i-1)*len(index)/3))+"~"+str(int(i*len(index)/3))+")")

    
    if pattern!=2:
      save_path = 'data/gcng_input/exp_matrix/pattern' + str(pattern) + '/'
    else:
      save_path = 'data/gcng_input/exp_matrix/pattern' + str(pattern) + 'a/'
    if not os.path.isdir(save_path):
      os.makedirs(save_path)

    
    np.save(save_path + str(i) + '_train_X.npy', np.array(X_train))
    np.save(save_path + str(i) + '_train_y.npy', np.array(y_train))
    np.save(save_path + str(i) + '_test_X.npy', np.array(X_test))
    np.save(save_path + str(i) + '_test_y.npy', np.array(y_test))
    np.save(save_path + str(i) + '_train_gene_list.npy', np.array(train_pair_list))
    np.save(save_path + str(i) + '_test_gene_list.npy', np.array(test_pair_list))
    
  



  # 評価用データセット2bを作成
  if pattern==2:
    print("[dataset "+str(pattern)+"b]")

    positive_gene_list=[]
    for k in range(len(positive_pairs)):
      positive_gene_list.append(positive_pairs[k][0])
      positive_gene_list.append(positive_pairs[k][1])

    positive_gene_list = list(set(positive_gene_list))
    
    edge_list = []
    indexes = [i for i in range(len(positive_gene_list))]
    for k in range(len(positive_pairs)):
      l = positive_gene_list.index(positive_pairs[k][0])
      m = positive_gene_list.index(positive_pairs[k][1])
      edge_list.append((indexes[l], indexes[m]))
      
    

    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, k=0.7)
    fig = plt.figure(figsize=(55,55))
    nx.draw_networkx(G, pos=pos, node_color="b")
    save_path = 'data/gcng_input/exp_matrix/pattern2b/image/'
    if not os.path.isdir(save_path):
      os.makedirs(save_path)
    fig.savefig(save_path+"whole.pdf")
    plt.close()

    # 連結成分ごとに分割
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    for i in range (len(S)):
      fig = plt.figure(figsize=(55,55))
      name = "S"+str(i)+".pdf"
      pos = nx.spring_layout(S[i], k=0.7)
      nx.draw_networkx(S[i], pos=pos, node_color="b")
      fig.savefig(save_path+name)
      plt.close()
    
    

    # 結果をまとめる (2分割交差検証用のデータ)
    pair_1 = []
    pair_2 = []
    edge_num_1=0
    edge_num_2=0

    for l in range(len(S)):
      if edge_num_1<edge_num_2:
        edge_num_1 = edge_num_1 + len(S[l].edges())
        for e in S[l].edges:
          pair_1.append([positive_gene_list[indexes.index(e[0])], positive_gene_list[indexes.index(e[1])]])
      else:
        edge_num_2 = edge_num_2 + len(S[l].edges())
        for e in S[l].edges:
          pair_2.append([positive_gene_list[indexes.index(e[0])], positive_gene_list[indexes.index(e[1])]])

    # ネガティブを作る
    negative_pairs_1 = []
    for i in range(len(pair_1)):
      while(1):
        pair = tuple(sorted(random.choices(all_gene,k=2)))
        if not pair in all_pair:
          negative_pairs_1.append(pair)
          break
    negative_pairs_2 = []
    for i in range(len(pair_2)):
      while(1):
        pair = tuple(sorted(random.choices(all_gene,k=2)))
        if not pair in all_pair:
          if not pair in negative_pairs_1:
            negative_pairs_2.append(pair)
            break

    save_path = 'data/gcng_input/exp_matrix/pattern2b/'
    if not os.path.isdir(save_path):
      os.makedirs(save_path)
    
    X_1 = []
    y_1 = []
    genes_list_1 = []
    X_2 = []
    y_2 = []
    genes_list_2 = []

    for j in range(len(pair_1)):
      X_1.append(np.array(counts[[pair_1[j][0], pair_1[j][1]]]))
      X_1.append(np.array(counts[[negative_pairs_1[j][0], negative_pairs_1[j][1]]]))
      y_1.append(1)
      y_1.append(0)
      genes_list_1.append(pair_1[j])
      genes_list_1.append(negative_pairs_1[j])
      
    for j in range(len(pair_2)):
      X_2.append(np.array(counts[[pair_2[j][0], pair_2[j][1]]]))
      X_2.append(np.array(counts[[negative_pairs_2[j][0], negative_pairs_2[j][1]]]))
      y_2.append(1)
      y_2.append(0)
      genes_list_2.append(pair_2[j])
      genes_list_2.append(negative_pairs_2[j])

    print("  <1> train: ", len(X_1), ", test: ", len(X_2))
    print("  <2> train: ", len(X_2), ", test: ", len(X_1))

    np.save(save_path + '1_train_X.npy', np.array(X_1))
    np.save(save_path + '1_train_y.npy', np.array(y_1))
    np.save(save_path + '1_train_gene_list.npy', np.array(genes_list_1))
    np.save(save_path + '1_test_X.npy', np.array(X_2))
    np.save(save_path + '1_test_y.npy', np.array(y_2))
    np.save(save_path + '1_test_gene_list.npy', np.array(genes_list_2))

    np.save(save_path + '2_train_X.npy', np.array(X_2))
    np.save(save_path + '2_train_y.npy', np.array(y_2))
    np.save(save_path + '2_train_gene_list.npy', np.array(genes_list_2))
    np.save(save_path + '2_test_X.npy', np.array(X_1))
    np.save(save_path + '2_test_y.npy', np.array(y_1))
    np.save(save_path + '2_test_gene_list.npy', np.array(genes_list_1))

        
    """
    # pair1とpair2で同じ遺伝子が含まれないか確認
    g1=[]
    g2=[]
    for i in range(len(pair_1)):
      g1.append(pair_1[i][0])
      g1.append(pair_1[i][1])
    for i in range(len(pair_2)):
      g2.append(pair_2[i][0])
      g2.append(pair_2[i][1])
    if len(list(set(g1) & set(g2))) != 0:
      print("errorrrr")

    # pair1とpair2が，きちんと分類されているかを確認
    if len(positive_pairs) != (len(pair_1)+len(pair_2)):
      print("Error")
    for pair in pair_1:
      if not (pair[0], pair[1]) in positive_pairs:
        if not (pair[1], pair[0]) in positive_pairs:
          print("Error")
    for pair in pair_2:
      if not (pair[0], pair[1]) in positive_pairs:
        if not (pair[1], pair[0]) in positive_pairs:
          print("Error")
    """
    