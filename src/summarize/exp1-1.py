import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import japanize_matplotlib


"""
実験1.1
  隣接行列2と、発現行列1,2,3
"""
if not os.path.isdir("result/Experiment1-1"):
  os.makedirs("result/Experiment1-1")

df_exp1  = pd.DataFrame(index=["1","2","3","mu","sd"], columns=["acc_SVM","acc_RandomForest","acc_GCN", "spe_SVM","spe_RandomForest","spe_GCN", "sen_SVM","sen_RandomForest","sen_GCN"])
df_exp2a = pd.DataFrame(index=["1","2","3","mu","sd"], columns=["acc_SVM","acc_RandomForest","acc_GCN", "spe_SVM","spe_RandomForest","spe_GCN", "sen_SVM","sen_RandomForest","sen_GCN"])
df_exp3  = pd.DataFrame(index=["1","2","3","mu","sd"], columns=["acc_SVM","acc_RandomForest","acc_GCN", "spe_SVM","spe_RandomForest","spe_GCN", "sen_SVM","sen_RandomForest","sen_GCN"])

result_svm_table = pd.read_table("data/Conventional_result/svm_result.csv", sep=',', header=0, index_col=0)
result_rf_table  = pd.read_table("data/Conventional_result/RandomForest_result.csv", sep=',', header=0, index_col=0)

for exp_pattern in ["1", "2a", "3"]:
  result_gcn_list = []
  result_svm_list = []
  result_rf_list  = []
  df_temp = pd.DataFrame(index=["1","2","3","mu","sd"], columns=["acc_SVM","acc_RandomForest","acc_GCN", "spe_SVM","spe_RandomForest","spe_GCN", "sen_SVM","sen_RandomForest","sen_GCN"])
  
  for i in range(1,4):
    result_gcng_table = pd.read_table("data/gcng_output/Laplacian/adj2-exp"+exp_pattern+"/result_test_data.csv", sep=',', header=0, index_col=0)
    df_temp.loc[str(i), 'acc_GCN'] = float(result_gcng_table.loc[i,"acc_last"])
    df_temp.loc[str(i), 'spe_GCN'] = float(result_gcng_table.loc[i,"Specificity_last"])
    df_temp.loc[str(i), 'sen_GCN'] = float(result_gcng_table.loc[i,"Sensitivity_last"])

    row_svm = int(result_svm_table.index.get_loc(exp_pattern))+i-1 
    df_temp.loc[str(i), 'acc_SVM'] = float(result_svm_table.iloc[row_svm, result_svm_table.columns.get_loc('acc')])
    df_temp.loc[str(i), 'spe_SVM'] = float(result_svm_table.iloc[row_svm, result_svm_table.columns.get_loc('Specificity')])
    df_temp.loc[str(i), 'sen_SVM'] = float(result_svm_table.iloc[row_svm, result_svm_table.columns.get_loc('Sensitivity')])

    row_rf = int(result_rf_table.index.get_loc(exp_pattern))+i-1
    df_temp.loc[str(i), 'acc_RandomForest'] = float(result_rf_table.iloc[row_rf, result_rf_table.columns.get_loc('acc')])
    df_temp.loc[str(i), 'spe_RandomForest'] = float(result_rf_table.iloc[row_rf, result_rf_table.columns.get_loc('Specificity')])
    df_temp.loc[str(i), 'sen_RandomForest'] = float(result_rf_table.iloc[row_rf, result_rf_table.columns.get_loc('Sensitivity')])
  
  if exp_pattern=="1":
    df_exp1  = df_temp
  elif exp_pattern=="2a":
    df_exp2a = df_temp
  elif exp_pattern=="3":
    df_exp3  = df_temp

x = np.array(['SVM', 'RandomForest', 'GCN'])
x_position = np.arange(len(x))


for i in range(9):
  l_1=[]
  l_2=[]
  l_3=[]
  for j in range(3):
    l_1.append(df_exp1.iloc[j,i])
    l_2.append(df_exp2a.iloc[j,i])
    l_3.append(df_exp3.iloc[j,i])
  df_exp1.iloc[3,i] = np.array(l_1).mean()
  df_exp1.iloc[4,i] = np.array(l_1).std()
  df_exp2a.iloc[3,i] = np.array(l_2).mean()
  df_exp2a.iloc[4,i] = np.array(l_2).std()
  df_exp3.iloc[3,i] = np.array(l_3).mean()
  df_exp3.iloc[4,i] = np.array(l_3).std()



error_bar_set = dict(lw = 1, capthick = 1, capsize = 20)
if not os.path.isdir("result/Experiment1-1"):
  os.makedirs("result/Experiment1-1")

plt.rcParams['font.family'] = 'IPAPGothic'
japanize_matplotlib.japanize()

for eva in ["Accuracy", "Specificity", "Sensitivity"]:
  plt.title('Experiment1-1 ('+eva+')')

  if eva=="Accuracy":
    y_1 = np.array(df_exp1.loc['mu', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    e_1 = np.array(df_exp1.loc['sd', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    y_2 = np.array(df_exp2a.loc['mu', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    e_2 = np.array(df_exp2a.loc['sd', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    y_3 = np.array(df_exp3.loc['mu', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
    e_3 = np.array(df_exp3.loc['sd', ['acc_SVM', 'acc_RandomForest', 'acc_GCN']])
  elif eva=="Specificity":
    y_1 = np.array(df_exp1.loc['mu', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    e_1 = np.array(df_exp1.loc['sd', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    y_2 = np.array(df_exp2a.loc['mu', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    e_2 = np.array(df_exp2a.loc['sd', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    y_3 = np.array(df_exp3.loc['mu', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
    e_3 = np.array(df_exp3.loc['sd', ['spe_SVM', 'spe_RandomForest', 'spe_GCN']])
  elif eva=="Sensitivity":
    y_1 = np.array(df_exp1.loc['mu', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    e_1 = np.array(df_exp1.loc['sd', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    y_2 = np.array(df_exp2a.loc['mu', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    e_2 = np.array(df_exp2a.loc['sd', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    y_3 = np.array(df_exp3.loc['mu', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])
    e_3 = np.array(df_exp3.loc['sd', ['sen_SVM', 'sen_RandomForest', 'sen_GCN']])


  plt.bar(x_position, y_1, yerr=e_1,
       tick_label=x, width=0.3, label="評価用データセット1",
       error_kw=error_bar_set)
  
  plt.bar(x_position+0.3, y_2, yerr=e_2,
       tick_label=x, width=0.3, label="評価用データセット2.a",
       error_kw=error_bar_set)
  
  plt.bar(x_position+0.6, y_3, yerr=e_3,
       tick_label=x, width=0.3, label="評価用データセット3",
       error_kw=error_bar_set)
  
  plt.legend(loc='best')
  plt.savefig('result/Experiment1-1/'+eva+'.pdf')
  plt.clf()


df_exp1.to_csv("result/Experiment1-1/exp1.csv")
df_exp2a.to_csv("result/Experiment1-1/exp2a.csv")
df_exp3.to_csv("result/Experiment1-1/exp3.csv")

