import pandas as pd
import collections
import matplotlib.pyplot as plt
import os

pvalue=pd.read_table("data/cellphonedb_output/pvalues.txt")

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

print("delete same pair in pvalue (from ", len(all_pair), " to ", len(list(set(all_pair))), ")")
all_pair = list(set(all_pair))


### 重複している遺伝子ペアは全て同じ同じp値になっていることを確認 → 一つにまとめてOK
"""
duplication = [k for k, v in collections.Counter(all_pair).items() if v > 1]
for a,b in duplication:
  index_list = []
  for i in range(len(pvalue)):
    if pvalue.loc[i, "gene_a"]==a and pvalue.loc[i, "gene_b"]==b:
      index_list.append(i)
  temp = pvalue.loc[index_list[0], "A|A":"I|I"].tolist()
  for j in index_list:
    if not temp==pvalue.loc[j, "A|A":"I|I"].tolist():
      print("Error")
      print(a, b, index_list)
    temp = pvalue.loc[j, "A|A":"I|I"].tolist()
"""


"""
評価用データセット用の正例ペアを決定 (pvalueにおけるインデックスを保存)
"""
positive_index_1 = []   # 評価用データセット1
for a,b in all_pair:
  for i in range(len(pvalue)):
    if pvalue.loc[i, 'gene_a']==a and pvalue.loc[i, 'gene_b']==b:
      positive_index_1.append(i)
      break

positive_index_2 = [] # 評価用データセット2a, 2b
positive_index_3 = [] # 評価用データセット3
for ind in positive_index_1:
  count = 0
  for j in pvalue.loc[ind, 'A|A':'I|I']:
    if j<=0.05:
      count = count+1
  if count>1:
    positive_index_2.append(ind)
  if count>17:
    positive_index_3.append(ind)

"""
結果を保存
"""
save_path = 'data/gcng_input/interaction_list/'
if not os.path.isdir(save_path):
  os.makedirs(save_path)

print("dataset 1 >> positive_num : ", len(positive_index_1))
print("dataset 2 >> positive_num : ", len(positive_index_2))
print("dataset 3 >> positive_num : ", len(positive_index_3))

with open(save_path+'positive_1.txt', 'w') as f:
  f.writelines(str(positive_index_1))

with open(save_path+'positive_2.txt', 'w') as f:
  f.writelines(str(positive_index_2))

with open(save_path+'positive_3.txt', 'w') as f:
  f.writelines(str(positive_index_3))


