import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

pattern_list = ["1111", "0000"]

normalized_method = "Laplacian"
#normalized_method = "endocrine"

exp_pattern = "exp2a"
#exp_pattern = "exp2b"
#exp_pattern = "exp3"


"""
  散布図をプロット
"""
for pattern in pattern_list:
  df_feature = pd.read_table("result/consideration/feature/"+normalized_method+"_"+exp_pattern+"/"+pattern+".csv",sep=',', index_col=0)
  corr = df_feature["corr"].tolist()
  moran_a = df_feature["moran_a"].tolist()
  moran_b = df_feature["moran_b"].tolist()
  plt.scatter(moran_a, moran_b)

plt.show()




"""
feature_names = ["corr", "sum_a", "sum_b"]

arrow_mul = 1
text_mul  = 1.1


X = []
target = []

temp=0
for pattern in pattern_list:
  df_feature = pd.read_table("result/consideration/feature/"+pattern+".csv",sep=',', index_col=0)[feature_names]
  X.extend(df_feature.to_numpy())
  target.extend([temp]*len(df_feature))
  temp = temp+1

ss = StandardScaler()
X = ss.fit_transform(X)
target = np.array(target)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

x_data = X[:,0]
y_data = X[:,1]

pc0 = pca.components_[0]
pc1 = pca.components_[1]


plt.figure()
plt.scatter(x_data, y_data, c=target/len(set(target)), marker=".")

for i in range(pc0.shape[0]):
  plt.arrow(0, 0, pc0[i]*arrow_mul, pc1[i]*arrow_mul, color='r')
  plt.text(pc0[i]*arrow_mul*text_mul, pc1[i]*arrow_mul*text_mul, feature_names[i], color='r')
  plt.show()
"""





