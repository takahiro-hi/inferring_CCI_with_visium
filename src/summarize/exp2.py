import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import csv

"""
実験2
  4パターンの隣接行列と、普通の発現行列
"""

result_adj1 = []
result_adj2 = []
result_adj3 = []
result_adj4 = []

if not os.path.isdir("result/Experiment2"):
  os.makedirs("result/Experiment2")


result_adj1_gcn_table = pd.read_table("data/gcng_output/Laplacian/adj1-exp2a/result_test_data.csv", sep=',', header=0, index_col=0)
result_adj2_gcn_table = pd.read_table("data/gcng_output/Laplacian/adj2-exp2a/result_test_data.csv", sep=',', header=0, index_col=0)
result_adj3_gcn_table = pd.read_table("data/gcng_output/Laplacian/adj3-exp2a/result_test_data.csv", sep=',', header=0, index_col=0)
result_adj4_gcn_table = pd.read_table("data/gcng_output/Laplacian/adj4-exp2a/result_test_data.csv", sep=',', header=0, index_col=0)

for i in range(1,4):
  result_adj1.append([float(result_adj1_gcn_table.loc[i,'acc_last']), float(result_adj1_gcn_table.loc[i,'Specificity_last']), float(result_adj1_gcn_table.loc[i,'Sensitivity_last'])])
  result_adj2.append([float(result_adj2_gcn_table.loc[i,'acc_last']), float(result_adj2_gcn_table.loc[i,'Specificity_last']), float(result_adj2_gcn_table.loc[i,'Sensitivity_last'])])
  result_adj3.append([float(result_adj3_gcn_table.loc[i,'acc_last']), float(result_adj3_gcn_table.loc[i,'Specificity_last']), float(result_adj3_gcn_table.loc[i,'Sensitivity_last'])])
  result_adj4.append([float(result_adj4_gcn_table.loc[i,'acc_last']), float(result_adj4_gcn_table.loc[i,'Specificity_last']), float(result_adj4_gcn_table.loc[i,'Sensitivity_last'])])


result_adj1_mean = np.mean(result_adj1, axis=0)
result_adj2_mean = np.mean(result_adj2, axis=0)
result_adj3_mean = np.mean(result_adj3, axis=0)
result_adj4_mean = np.mean(result_adj4, axis=0)

result_adj1_std = np.std(result_adj1, axis=0)
result_adj2_std = np.std(result_adj2, axis=0)
result_adj3_std = np.std(result_adj3, axis=0)
result_adj4_std = np.std(result_adj4, axis=0)

"""
print("adj1")
print(result_adj1)
print("adj2")
print(result_adj2)
print("adj3")
print(result_adj3)
print("adj4")
print(result_adj4)
"""


x = np.array(['adj1', 'adj2', 'adj3', 'adj4'])
x_position = np.arange(len(x))




for eva in ["Accuracy", "Specificity", "Sensitivity"]:
  plt.rcParams['font.family'] = 'IPAPGothic'
  japanize_matplotlib.japanize()

  if eva=="Accuracy":
    y = np.array([result_adj1_mean[0], result_adj2_mean[0], result_adj3_mean[0], result_adj4_mean[0]])
    e = np.array([result_adj1_std[0], result_adj2_std[0], result_adj3_std[0], result_adj4_std[0]])
  elif eva=="Specificity":
    y = np.array([result_adj1_mean[1], result_adj2_mean[1], result_adj3_mean[1], result_adj4_mean[1]])
    e = np.array([result_adj1_std[1], result_adj2_std[1], result_adj3_std[1], result_adj4_std[1]])
  elif eva=="Sensitivity":
    y = np.array([result_adj1_mean[2], result_adj2_mean[2], result_adj3_mean[2], result_adj4_mean[2]])
    e = np.array([result_adj1_std[2], result_adj2_std[2], result_adj3_std[2], result_adj4_std[2]])

  error_bar_set = dict(lw = 1, capthick = 1, capsize = 20)


  plt.title('Experiment2 ('+eva+')')

  plt.bar(x_position, y, yerr = e,
       tick_label=x,
       error_kw=error_bar_set)
  #plt.show()
  plt.savefig('result/Experiment2/'+eva+'.pdf')
  plt.clf()



with open("result/Experiment2/result.csv",'w', newline='') as f:
  dataWriter = csv.writer(f)
  dataWriter.writerow([" ", "adj1", "adj2", "adj3", "adj4"])
  dataWriter.writerow(["Accuracy", result_adj1_mean[0], result_adj2_mean[0], result_adj3_mean[0], result_adj4_mean[0]])
  dataWriter.writerow(["Specificity", result_adj1_mean[1], result_adj2_mean[1], result_adj3_mean[1], result_adj4_mean[1]])
  dataWriter.writerow(["Sensitivity", result_adj1_mean[2], result_adj2_mean[2], result_adj3_mean[2], result_adj4_mean[2]])
