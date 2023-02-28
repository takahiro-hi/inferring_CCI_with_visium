import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


save_path = "result/consideration/heatmap/"
if not os.path.isdir(save_path):
  os.makedirs(save_path)


"""
  ヒートマップをまとめて，予測パターンごとにクラスタリングを行う
"""
exp_pattern = "2a"

# ヒートマップのデータをロードし，一つにまとめる
df_heatmap_whole_positive = pd.DataFrame(columns=["adj1","adj2","adj3", "adj4", "sum"])

for adj_pattern in ["adj1", "adj2", "adj3", "adj4"]:
  df_heatmap_positive = pd.read_table("data/gcng_output/Laplacian/"+adj_pattern+"-exp2a/heatmap_positive.csv", sep=',', index_col=0)[adj_pattern]
  df_heatmap_whole_positive[adj_pattern] = df_heatmap_positive


# 各隣接パターンでの予測パターンをクラスタリングする
sum_list=[]
for i in range(len(df_heatmap_whole_positive)):
  num = df_heatmap_whole_positive.iloc[i, 0:4].tolist()
  sum1 = int(1000*num[0]+100*num[1]+10*num[2]+1*num[3])
  sum_list.append(sum1)
df_heatmap_whole_positive["sum"] = sum_list

df_heatmap_whole_positive = df_heatmap_whole_positive.sort_values('sum', ascending=False)
df_heatmap_whole_positive.to_csv(save_path+"positive.csv", index=True)


# ヒートマップをプロット
plt.figure(figsize=(20, 20))
sns.set(font_scale = 5)
df_heatmap_whole_positive = df_heatmap_whole_positive.drop(columns="sum", axis=1)
sns.heatmap(df_heatmap_whole_positive, xticklabels=True, yticklabels=True, cmap='Reds', cbar=False) #, linewidths=0.5)
plt.savefig(save_path+"positive.pdf")
plt.clf()
