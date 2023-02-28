import matplotlib.pyplot as plt
import parameters
import numpy as np
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalseNegatives, FalsePositives
import os
import pandas as pd

"""
ヒートマップ用のデータフレームを返す
"""
def return_df_heat(normalized_pattern, exp_pattern, adj_pattern):
  if os.path.isfile('data/gcng_output/'+normalized_pattern+'/adj'+adj_pattern+'-exp'+exp_pattern+'/heatmap_negative.csv'):
    df_heat_ne = pd.read_csv('data/gcng_output/'+normalized_pattern+'/adj'+adj_pattern+'-exp'+exp_pattern+'/heatmap_negative.csv', index_col=0)
    df_heat_po = pd.read_csv('data/gcng_output/'+normalized_pattern+'/adj'+adj_pattern+'-exp'+exp_pattern+'/heatmap_positive.csv', index_col=0)
  else:
    pair_po=[]
    pair_ne=[]
    if exp_pattern!='2b':
      for i in [1,2,3]:
        pair_list = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(i) + '_test_gene_list.npy')
        y_test    = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(i) + '_test_y.npy')
        for j in range(len(pair_list)):
          if y_test[j]==1:
            pair_po.append(pair_list[j][0]+'-'+pair_list[j][1])
          elif y_test[j]==0:
            pair_ne.append(pair_list[j][0]+'-'+pair_list[j][1])
    else:
      for i in [1,2]:
        pair_list = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(i) + '_test_gene_list.npy')
        y_test    = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(i) + '_test_y.npy')
        for j in range(len(pair_list)):
          if y_test[j]==1:
            pair_po.append(pair_list[j][0]+'-'+pair_list[j][1])
          elif y_test[j]==0:
            pair_ne.append(pair_list[j][0]+'-'+pair_list[j][1])

    heat_po = np.zeros((len(pair_po), 4))
    heat_ne = np.zeros((len(pair_ne), 4))

    df_heat_po = pd.DataFrame(heat_po,index=pair_po,columns=["adj1","adj2","adj3","adj4"])
    df_heat_ne = pd.DataFrame(heat_ne,index=pair_ne,columns=["adj1","adj2","adj3","adj4"])

  return df_heat_po, df_heat_ne

"""
ヒートマップ用のデータフレームを更新
"""
def updata_df_heat(df_heat_po, df_heat_ne, pre_te, y_test, gene_pair_test, adj_pattern):
  for i in range(len(pre_te)):
    if y_test[i]==1:
      if pre_te[i]>=0.5:
        df_heat_po.loc[gene_pair_test[i][0]+'-'+gene_pair_test[i][1], "adj"+str(adj_pattern)] = 1
      else:
        df_heat_po.loc[gene_pair_test[i][0]+'-'+gene_pair_test[i][1], "adj"+str(adj_pattern)] = 0
    elif y_test[i]==0:
      if pre_te[i]>=0.5:
        df_heat_ne.loc[gene_pair_test[i][0]+'-'+gene_pair_test[i][1], "adj"+str(adj_pattern)] = -1
      else:
        df_heat_ne.loc[gene_pair_test[i][0]+'-'+gene_pair_test[i][1], "adj"+str(adj_pattern)] = 0
 
  return df_heat_po, df_heat_ne

"""
SensitivityとSpecificityを計算
"""
def calc_sen_spe(target, predictions):
  tp = TruePositives()
  tp.reset_state()
  tp.update_state(target, predictions)
  tp_val = tp.result().numpy()

  tn = TrueNegatives()
  tn.reset_state()
  tn.update_state(target, predictions)
  tn_val = tn.result().numpy()

  fn = FalseNegatives()
  fn.reset_state()
  fn.update_state(target, predictions)
  fn_val = fn.result().numpy()

  fp = FalsePositives()
  fp.reset_state()
  fp.update_state(target, predictions)
  fp_val = fp.result().numpy()

  Sensitivity = tp_val / (tp_val+fn_val)
  Specificity = tn_val / (tn_val+fp_val)

  return Sensitivity, Specificity


"""
渡されたリストの値をグラフにプロットして保存する
"""
def plot_list(val_list, name, save_path, test_indel):
  fig = plt.figure(figsize=(13,16))
  epoch = [i for i in range(0,parameters.num_epoch)]
  val_list = np.array(val_list).T

  plt.title(name)
  plt.xlabel("Epochs")
  plt.ylabel(name)

  for i, array in enumerate(val_list):
    if i==0:
      label = "train data"
      color = "b"
    elif i==1:
      label = "validation data"
      color = "g"
    elif i==2:
      label = "test data"
      color = "r"

    plt.plot(epoch, array, color=color, marker=".", label=label, alpha=0.5)

  plt.legend()
  fig.savefig(save_path+"/"+str(test_indel)+'/' + name + '.png')




