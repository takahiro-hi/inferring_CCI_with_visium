import pandas as pd
import os

if not os.path.isdir("data/cellphonedb_input"):
        os.makedirs("data/cellphonedb_input")


"""
クラスター情報
  ・txt形式
  ・タブ区切り
  ・クラスタ情報をA~Iにエンコード
"""
clusters = pd.read_table("data/visium_data/clusters.csv", index_col="Barcode", sep=',')
list = "ABCDEFGHI"

for i in range(len(clusters)):
  clusters.iloc[i] = list[int(clusters.iloc[i]["Cluster"])-1]

clusters.to_csv("data/cellphonedb_input/clusters_reshaped.txt", sep='\t')


"""
カウント情報
  ・タブ区切り
"""
counts=pd.read_table("data/visium_data/Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix_ENSID.txt", sep=" ", index_col="GENE")
counts.to_csv("data/cellphonedb_input/counts_reshaped.txt", sep='\t')


