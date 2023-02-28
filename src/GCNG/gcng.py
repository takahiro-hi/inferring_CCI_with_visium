import numpy as np
import os
import pickle
import csv
import pandas as pd
import seaborn as sns
import tensorflow as tf
from spektral.utils.sparse import sp_matrix_to_sp_tensor
import argparse
from scipy import sparse as sp
import matplotlib.pyplot as plt
import parameters
import model_gcn
import methods_gcng as methods


#import warnings
#warnings.filterwarnings('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

parser = argparse.ArgumentParser()
parser.add_argument('adj_pattern')
parser.add_argument('exp_pattern')
parser.add_argument('cv_number')
parser.add_argument('normalized')   # "autocrine", "endocrine", "Laplacian", "normal"
args = parser.parse_args()

save_path = 'data/gcng_output/'+args.normalized+'/adj'+args.adj_pattern+'-exp'+args.exp_pattern+'/'
if not os.path.isdir(save_path):
  os.makedirs(save_path)

start = int(args.cv_number)
end = start+1

adj_name = 'data/gcng_input/adj_matrix/adj'+args.adj_pattern+'/'+args.normalized
with open(adj_name, 'rb') as fp:
  adj = pickle.load(fp)
adj = sp_matrix_to_sp_tensor(adj)

# ヒートマップ用のデータフレームをロード
df_heat_po, df_heat_ne = methods.return_df_heat(normalized_pattern=args.normalized, exp_pattern=args.exp_pattern, adj_pattern=args.adj_pattern)


for test_indel in range(start,end):
  """
  データをロード・整形
  """
  X_data_train = np.load("data/gcng_input/exp_matrix/pattern" + args.exp_pattern + "/" + str(test_indel) + '_train_X.npy')
  y_data_train = np.load("data/gcng_input/exp_matrix/pattern" + args.exp_pattern + "/" + str(test_indel) + '_train_y.npy')
  X_data_test = np.load("data/gcng_input/exp_matrix/pattern" + args.exp_pattern + "/" + str(test_indel) + '_test_X.npy')
  y_data_test = np.load("data/gcng_input/exp_matrix/pattern" + args.exp_pattern + "/" + str(test_indel) + '_test_y.npy')
  gene_pair_test = np.load("data/gcng_input/exp_matrix/pattern" + args.exp_pattern + "/" + str(test_indel) + '_test_gene_list.npy')

  index = [i for i in range(y_data_train.shape[0])]
  validation_index = index[:int(np.ceil(0.3*len(index)))]
  train_index = index[int(np.ceil(0.3*len(index))):]

  X_train = X_data_train[train_index]
  X_val   = X_data_train[validation_index]
  X_test  = X_data_test
  y_train = y_data_train[train_index][:,np.newaxis]
  y_val   = y_data_train[validation_index][:,np.newaxis]
  y_test  = y_data_test[:,np.newaxis]


  """
  モデルの構築と学習
  """
  if args.exp_pattern=="2b":
    print("\n### ", test_indel, "/ 2 ### ", X_train.shape, X_val.shape, X_test.shape)
  else:
    print("\n### ", test_indel, "/ 3 ### ", X_train.shape, X_val.shape, X_test.shape)

  model = model_gcn.Net()

  num_data = X_train.shape[0]
  results_loss = []
  results_acc = []
  results_Specificity = []
  results_Sensitivity = []

  # モデル保存のタイミングを判定するための変数
  acc_temp, acc_temp_last = 0,0
  
  for epoch in range(parameters.num_epoch):
    # モデルの学習
    sff_idx = np.random.permutation(num_data)
    for idx in range(0, num_data, parameters.batch_size):
        if num_data < idx + parameters.batch_size:
          break
        batch_x = X_train[sff_idx[idx: idx + parameters.batch_size]]
        batch_t = y_train[sff_idx[idx: idx + parameters.batch_size]]
        inputs = (batch_x, adj)
        target = batch_t

        loss_tr, pre_tr = model_gcn.train_on_batch(inputs, target, model)

    loss_tr, acc_tr, pre_tr, sensi_tr, speci_tr = model_gcn.evaluate(((X_train, adj), y_train), model)
    loss_va, acc_va, pre_va, sensi_va, speci_va = model_gcn.evaluate(((X_val, adj), y_val), model)
    loss_te, acc_te, pre_te, sensi_te, speci_te = model_gcn.evaluate(((X_test, adj), y_test), model)


    # 結果をリストに保存
    results_loss.append((loss_tr, loss_va, loss_te))
    results_acc.append((acc_tr, acc_va, acc_te))
    results_Sensitivity.append((sensi_tr, sensi_va, sensi_te))
    results_Specificity.append((speci_tr, speci_va, speci_te))

    # モデルの保存
    if parameters.first_epoch < epoch:
      if acc_temp < acc_va:
        acc_temp = acc_va
        print(
          "Epoch >> {} | Train loss/acc : {:.4f}/{:.4f}, | Valid loss/acc : {:.4f}/{:.4f}, | Test loss/acc : {:.4f}/{:.4f}".format(
            epoch, loss_tr, acc_tr, loss_va, acc_va, loss_te, acc_te
          )
        )
        tf.saved_model.save(model, save_path+"/"+str(test_indel)+"/model")
        # 最終的な結果となる値を保存
        result_list = [round(float(acc_te),4), round(float(loss_te),4), speci_te, sensi_te]
        # ヒートマップ用データフレームを更新
        #df_heat_po, df_heat_ne = methods.updata_df_heat(df_heat_po, df_heat_ne, pre_te, y_test, gene_pair_test, args.adj_pattern)
        
    if parameters.last_epoch < epoch:
      if acc_temp_last < acc_va:
        acc_temp_last = acc_va
        tf.saved_model.save(model, save_path+"/"+str(test_indel)+"/model_last")
        # 最終的な結果となる値を保存
        result_last_list = [round(float(acc_te),4), round(float(loss_te),4), speci_te, sensi_te]
        # ヒートマップ用データフレームを更新
        df_heat_po, df_heat_ne = methods.updata_df_heat(df_heat_po, df_heat_ne, pre_te, y_test, gene_pair_test, args.adj_pattern)



  """
  結果をまとめる
  """
  # 各値の学習過程をグラフにプロット
  methods.plot_list(results_loss, 'loss', save_path, test_indel)
  methods.plot_list(results_acc, 'Accuracy', save_path, test_indel)
  methods.plot_list(results_Specificity, 'Specificity', save_path, test_indel)
  methods.plot_list(results_Sensitivity, 'Sensitivity', save_path, test_indel)

  # 最終的な結果をまとめる
  if os.path.isfile(save_path+"/result_test_data.csv"):
    df_result = pd.read_csv(save_path+"/result_test_data.csv", header=0, index_col=0)
    df_result.loc[test_indel] = [result_list[0], result_list[1], result_list[2], result_list[3], result_last_list[0], result_last_list[1], result_last_list[2], result_last_list[3]]
    df_result.to_csv(save_path+"/result_test_data.csv")
  else:
    with open(save_path+"/result_test_data.csv",'w', newline='') as f:
      dataWriter = csv.writer(f)
      dataWriter.writerow([" ", "acc", "loss", "Specificity", "Sensitivity", "acc_last", "loss_last", "Specificity_last", "Sensitivity_last"])
      dataWriter.writerow([test_indel, result_list[0], result_list[1], result_list[2], result_list[3], result_last_list[0], result_last_list[1], result_last_list[2], result_last_list[3]])

  # 学習のログを保存
  results_loss = np.array(results_loss).T
  results_acc  = np.array(results_acc).T
  df_log = pd.DataFrame()
  df_log['train_loss'] = results_loss[0]
  df_log['train_acc']  = results_acc[0]
  df_log['valid_loss'] = results_loss[1]
  df_log['valid_acc']  = results_acc[1]
  df_log['test_loss']  = results_loss[2]
  df_log['test_acc']   = results_acc[2]
  df_log.to_csv(save_path+"/"+str(test_indel)+"/log.csv")

  # ヒートマップを作成，データも保存
  df_heat_po.to_csv(save_path+'/heatmap_positive.csv')
  df_heat_ne.to_csv(save_path+'/heatmap_negative.csv')

  plt.figure(figsize=(20, 20))
  sns.heatmap(df_heat_po, xticklabels=True, yticklabels=True, cmap='Reds') #, linewidths=0.5)
  plt.savefig(save_path+'/heatmap_positive.pdf')
  plt.clf()

  plt.figure(figsize=(20, 20))
  sns.heatmap(df_heat_ne, xticklabels=True, yticklabels=True, cmap='Blues') #, linewidths=0.5)
  plt.savefig(save_path+'/heatmap_negative.pdf')
  plt.clf()










