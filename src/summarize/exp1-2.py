import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import japanize_matplotlib

normalized_method = "Laplacian"
#normalized_method = "endocrine"


"""
実験1.2
  隣接行列2と、発現行列2a,2b
"""

if not os.path.isdir("result/Experiment1-2/"+normalized_method):
  os.makedirs("result/Experiment1-2/"+normalized_method)


result_2a_gcn_table = pd.read_table("data/gcng_output/"+normalized_method+"/adj2-exp2a/result_test_data.csv", sep=',', header=0, index_col=0)
result_2b_gcn_table = pd.read_table("data/gcng_output/"+normalized_method+"/adj2-exp2b/result_test_data.csv", sep=',', header=0, index_col=0)
result_svm_table = pd.read_table("data/Conventional_result/svm_result.csv", sep=',', header=0, index_col=0)
result_rf_table  = pd.read_table("data/Conventional_result/RandomForest_result.csv", sep=',', header=0, index_col=0)


df_2a = pd.DataFrame(index=["1","2","3","mu","sd"], columns=["acc_SVM","acc_RandomForest","acc_GCN", "spe_SVM","spe_RandomForest","spe_GCN", "sen_SVM","sen_RandomForest","sen_GCN"])
df_2b = pd.DataFrame(index=["1","2","mu","sd"], columns=["acc_SVM","acc_RandomForest","acc_GCN", "spe_SVM","spe_RandomForest","spe_GCN", "sen_SVM","sen_RandomForest","sen_GCN"])

for i in range(1,4):
  df_2a.loc[str(i), "acc_GCN"] = float(result_2a_gcn_table.loc[i, "acc_last"])
  df_2a.loc[str(i), "spe_GCN"] = float(result_2a_gcn_table.loc[i, "Specificity_last"])
  df_2a.loc[str(i), "sen_GCN"] = float(result_2a_gcn_table.loc[i, "Sensitivity_last"])

for i in range(1,3):
  df_2b.loc[str(i), "acc_GCN"] = float(result_2b_gcn_table.loc[i, "acc_last"])
  df_2b.loc[str(i), "spe_GCN"] = float(result_2b_gcn_table.loc[i, "Specificity_last"])
  df_2b.loc[str(i), "sen_GCN"] = float(result_2b_gcn_table.loc[i, "Sensitivity_last"])

for i in range(1,4):
  row_svm_2a = int(result_svm_table.index.get_loc("2a"))+i-1
  row_rf_2a = int(result_rf_table.index.get_loc("2a"))+i-1
  df_2a.loc[str(i), 'acc_SVM'] = float(result_svm_table.iloc[row_svm_2a, result_svm_table.columns.get_loc('acc')])
  df_2a.loc[str(i), 'spe_SVM'] = float(result_svm_table.iloc[row_svm_2a, result_svm_table.columns.get_loc('Specificity')])
  df_2a.loc[str(i), 'sen_SVM'] = float(result_svm_table.iloc[row_svm_2a, result_svm_table.columns.get_loc('Sensitivity')])
  df_2a.loc[str(i), 'acc_RandomForest'] = float(result_rf_table.iloc[row_rf_2a, result_rf_table.columns.get_loc('acc')])
  df_2a.loc[str(i), 'spe_RandomForest'] = float(result_rf_table.iloc[row_rf_2a, result_rf_table.columns.get_loc('Specificity')])
  df_2a.loc[str(i), 'sen_RandomForest'] = float(result_rf_table.iloc[row_rf_2a, result_rf_table.columns.get_loc('Sensitivity')])
  
  row_svm_2b = int(result_svm_table.index.get_loc("2b"))+i-1
  row_rf_2b = int(result_rf_table.index.get_loc("2b"))+i-1
  df_2b.loc[str(i), 'acc_SVM'] = float(result_svm_table.iloc[row_svm_2b, result_svm_table.columns.get_loc('acc')])
  df_2b.loc[str(i), 'spe_SVM'] = float(result_svm_table.iloc[row_svm_2b, result_svm_table.columns.get_loc('Specificity')])
  df_2b.loc[str(i), 'sen_SVM'] = float(result_svm_table.iloc[row_svm_2b, result_svm_table.columns.get_loc('Sensitivity')])
  df_2b.loc[str(i), 'acc_RandomForest'] = float(result_rf_table.iloc[row_rf_2b, result_rf_table.columns.get_loc('acc')])
  df_2b.loc[str(i), 'spe_RandomForest'] = float(result_rf_table.iloc[row_rf_2b, result_rf_table.columns.get_loc('Specificity')])
  df_2b.loc[str(i), 'sen_RandomForest'] = float(result_rf_table.iloc[row_rf_2b, result_rf_table.columns.get_loc('Sensitivity')])




x = np.array(['SVM', 'RandomForest', 'GCN'])
x_position = np.arange(len(x))

for i in range(0,9):
  li_a = []
  li_b = []
  for j in range(0,3):
    li_a.append(df_2a.iloc[j,i])
  for j in range(0,2):
    li_b.append(df_2b.iloc[j,i])

  df_2a.iloc[3,i] = np.array(li_a).mean()
  df_2a.iloc[4,i] = np.array(li_a).std()
  df_2b.iloc[2,i] = np.array(li_b).mean()
  df_2b.iloc[3,i] = np.array(li_b).std()


error_bar_set = dict(lw = 1, capthick = 1, capsize = 20)


for eva in ["Accuracy", "Specificity", "Sensitivity"]:
  plt.rcParams['font.family'] = 'IPAPGothic'
  japanize_matplotlib.japanize()
  plt.title('Experiment1-2 ('+eva+')')
  if eva=="Accuracy":
    y_a = np.array(df_2a.loc['mu', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    e_a = np.array(df_2a.loc['sd', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    y_b = np.array(df_2b.loc['mu', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    e_b = np.array(df_2b.loc['sd', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
  elif eva=="Specificity":
    y_a = np.array(df_2a.loc['mu', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    e_a = np.array(df_2a.loc['sd', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    y_b = np.array(df_2b.loc['mu', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    e_b = np.array(df_2b.loc['sd', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
  elif eva=="Sensitivity":
    y_a = np.array(df_2a.loc['mu', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    e_a = np.array(df_2a.loc['sd', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    y_b = np.array(df_2b.loc['mu', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    e_b = np.array(df_2b.loc['sd', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])


  plt.bar(x_position, y_a, yerr=e_a,
       tick_label=x, width=0.3, label="gene_pair 2a", #"評価用データセット2a",
       error_kw=error_bar_set)
  
  plt.bar(x_position+0.3, y_b, yerr=e_b,
       tick_label=x, width=0.3, label="gene_pair 2b", #"評価用データセット2b",
       error_kw=error_bar_set)
  

  plt.legend(loc='best')
  #plt.show()

  plt.savefig('result/Experiment1-2/'+normalized_method+'/'+eva+'.png')
  plt.clf()


df_2a.to_csv("result/Experiment1-2/"+normalized_method+'/'+"exp2a.csv")
df_2b.to_csv("result/Experiment1-2/"+normalized_method+'/'+"exp2b.csv")
  
  