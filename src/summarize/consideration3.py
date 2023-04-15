import pandas as pd
import libpysal
from libpysal.weights import W
import numpy as np
from esda.moran import Moran
from esda.moran import Moran_BV_matrix
import os

normalized_method = "Laplacian"
#normalized_method = "endocrine"

exp_pattern = "exp2a"
#exp_pattern = "exp2b"
#exp_pattern = "exp3"


save_path = "result/consideration/feature/"+normalized_method+"_"+exp_pattern+"/"
if not os.path.isdir(save_path):
  os.makedirs(save_path)


df_heatmap = pd.read_table("result/consideration/heatmap/"+normalized_method+"/"+exp_pattern+"_positive.csv",sep=',', index_col=0)
spatial = pd.read_csv("data/visium_data/tissue_positions_list.csv", header=None)
clusters = pd.read_table("data/cellphonedb_input/clusters_reshaped.txt")
counts = pd.read_table("data/cellphonedb_input/counts_reshaped.txt", index_col="GENE").T

"""
  空間重み行列を作成
"""
all_spot_name = spatial.iloc[:,0].tolist()
spot_name = clusters.iloc[:,0].tolist()

number = []
for i in range(len(spot_name)):
  number.append(all_spot_name.index(spot_name[i]))

df_spatial=spatial.iloc[number, [0,4,5]].set_axis(['name','x','y'], axis=1)
kd = libpysal.cg.KDTree(np.array(df_spatial.loc[:,["x","y"]]))
wnn2 = libpysal.weights.KNN(kd, 18)


"""
モラン・相関係数などをデータフレームで保存
"""
# 計算した予測パターンを指定
pattern_list = ["1111","1110","1101","1011","0111","1100","1010","1001","0110","0101","0011","1000","0100","0010","0001","0000"]

# 指定した予測パターンに含まれる遺伝子ペアを保存
gene_pair_list = []
for pattern in pattern_list:
  gene_pair_temp = []
  for i in range(len(df_heatmap)):
    if df_heatmap.iloc[i]["sum"] == int(pattern):
      gene_pair_temp.append(df_heatmap.index.values[i])
  gene_pair_list.append(gene_pair_temp)


# モラン・相関係数などを計算
temp=0
for gene_list in gene_pair_list:
  df_feature = pd.DataFrame(columns=["corr","moran_a","moran_b","sum_a","sum_b", "moran_ave", "sum_ave"], index=gene_list)
  corr_list,moran_a_list,moran_b_list,moran_ab_list,sum_a_list,sum_b_list,moran_ave_list,sum_ave_list = [],[],[],[],[],[],[],[]
  for gene_pair in gene_list:
    genes = gene_pair.split('-')
    gene_a_counts = np.array(list(counts[genes[0]]))
    gene_b_counts = np.array(list(counts[genes[1]]))

    ## 相関係数を計算
    corr_list.append(np.corrcoef(gene_a_counts, gene_b_counts)[0,1])
    
    ## モランを計算
    moran_a_list.append(Moran(gene_a_counts, wnn2).I)
    moran_b_list.append(Moran(gene_b_counts, wnn2).I)
    moran_ave_list.append((Moran(gene_a_counts, wnn2).I + Moran(gene_b_counts, wnn2).I) / 2)
    moran_ab_list.append(Moran_BV_matrix([gene_a_counts, gene_b_counts],  wnn2, varnames=genes)[(0,  1)].I)

    # 発現の総量を計算
    sum_a_list.append(sum(list(gene_a_counts)))
    sum_b_list.append(sum(list(gene_b_counts)))
    sum_ave_list.append((sum(list(gene_a_counts)) + sum(list(gene_b_counts))) / 2)

  # データフレームに保存
  df_feature["corr"] = corr_list
  df_feature["moran_a"] = moran_a_list
  df_feature["moran_b"] = moran_b_list
  df_feature["moran_ab"] = moran_ab_list
  df_feature["sum_a"] = sum_a_list
  df_feature["sum_b"] = sum_b_list
  df_feature["moran_ave"] = moran_ave_list
  df_feature["sum_ave"] = sum_ave_list

  df_feature.to_csv(save_path+str(pattern_list[temp])+".csv", index=True)
  temp = temp+1

  







