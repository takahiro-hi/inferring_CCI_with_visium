from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalseNegatives, FalsePositives


"""
SpecificityとSensitivityを計算する関数
"""
def cal_result(target, predictions):
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
SVMとRandomForestを使用した結果を計算して，保存する
"""
for exp_pattern in ["1", "2a", "2b", "3"]:
  if exp_pattern=="2b":
    e = 3
  else:
    e = 4

  result_svm = []
  result_rf  = []
  
  print("## expression matrix : pattern " + exp_pattern + " ##")

  for cv in range(1,e):
    ## データをダウンロードして，一次元に変形
    X_data_train = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(cv) + '_train_X.npy')
    y_data_train = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(cv) + '_train_y.npy')
    X_data_test  = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(cv) + '_test_X.npy')
    y_data_test  = np.load("data/gcng_input/exp_matrix/pattern" + exp_pattern + "/" + str(cv) + '_test_y.npy')
  
    X_train_1 = []
    X_train_2 = []
    X_train_integrated = []
    X_test_1 = []
    X_test_2 = []
    X_test_integrated = []

    for i in range(X_data_train.shape[-3]):
      for j in range(X_data_train.shape[-2]):
        X_train_1.append(X_data_train[i,j,0])
        X_train_2.append(X_data_train[i,j,1])
      X_train_1.extend(X_train_2)
      X_train_integrated.append(X_train_1)
      X_train_1 = []
      X_train_2 = []

    for i in range(X_data_test.shape[-3]):
      for j in range(X_data_test.shape[-2]):
        X_test_1.append(X_data_test[i,j,0])
        X_test_2.append(X_data_test[i,j,1])
      X_test_1.extend(X_test_2)
      X_test_integrated.append(X_test_1)
      X_test_1 = []
      X_test_2 = []


    ## 従来手法で学習・予測
    clf_svm = svm.SVC(gamma="scale")
    clf_svm.fit(X_train_integrated, y_data_train)
    acc_svm = clf_svm.score(X_test_integrated, y_data_test)
    pre_svm = clf_svm.predict(X_test_integrated)
    pre_svm = np.array(pre_svm)
    Sensitivity_svm, Specificity_svm = cal_result(y_data_test, pre_svm)
    result_svm.append([acc_svm, Sensitivity_svm, Specificity_svm])


    clf_forest = RandomForestClassifier()
    clf_forest.fit(X_train_integrated, y_data_train)
    acc_random_forest = clf_forest.score(X_test_integrated, y_data_test)
    pre_random_forest = clf_forest.predict(X_test_integrated)
    pre_random_forest = np.array(pre_random_forest)
    Sensitivity_random_forest, Specificity_random_forest = cal_result(y_data_test, pre_random_forest)
    result_rf.append([acc_random_forest, Sensitivity_random_forest, Specificity_random_forest])

    print( " <{}> SVM/RF ... | Accuracy : {:.4f}/{:.4f} | Specificity : {:.4f}/{:.4f} | Sensitivity : {:.4f}/{:.4f}".format(
      str(cv), acc_svm, acc_random_forest, Specificity_svm, Specificity_random_forest, Sensitivity_svm, Sensitivity_random_forest
    ))

  
  save_path = 'data/Conventional_result/'
  if not os.path.isdir(save_path):
    os.makedirs(save_path)
  if os.path.isfile(save_path+"svm_result.csv"):
    with open(save_path+"svm_result.csv",'a', newline='') as f:
      dataWriter = csv.writer(f)
      for i in range(len(result_svm)):
        if i ==0:
          dataWriter.writerow([exp_pattern, str(i+1), result_svm[i][0], result_svm[i][1], result_svm[i][2]])
        else:
          dataWriter.writerow([" ", str(i+1), result_svm[i][0], result_svm[i][1], result_svm[i][2]])
  else:
    with open(save_path+"svm_result.csv",'w', newline='') as f:
      dataWriter = csv.writer(f)
      dataWriter.writerow(["exp_pattern", "cv", "acc", "Sensitivity", "Specificity"])
      for i in range(len(result_svm)):
        if i ==0:
          dataWriter.writerow([exp_pattern, str(i+1), result_svm[i][0], result_svm[i][1], result_svm[i][2]])
        else:
          dataWriter.writerow([" ", str(i+1), result_svm[i][0], result_svm[i][1], result_svm[i][2]])
  

  if os.path.isfile(save_path+"RandomForest_result.csv"):
    with open(save_path+"RandomForest_result.csv",'a', newline='') as f:
      dataWriter = csv.writer(f)
      for i in range(len(result_rf)):
        if i ==0:
          dataWriter.writerow([exp_pattern, str(i+1), result_rf[i][0], result_rf[i][1], result_rf[i][2]])
        else:
          dataWriter.writerow([" ", str(i+1), result_rf[i][0], result_rf[i][1], result_rf[i][2]])
  else:
    with open(save_path+"RandomForest_result.csv",'w', newline='') as f:
      dataWriter = csv.writer(f)
      dataWriter.writerow(["exp_pattern", "cv", "acc", "Sensitivity", "Specificity"])
      for i in range(len(result_rf)):
        if i ==0:
          dataWriter.writerow([exp_pattern, str(i+1), result_rf[i][0], result_rf[i][1], result_rf[i][2]])
        else:
          dataWriter.writerow([" ", str(i+1), result_rf[i][0], result_rf[i][1], result_rf[i][2]])









