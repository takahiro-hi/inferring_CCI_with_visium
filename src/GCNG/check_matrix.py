import numpy as np
from matplotlib_venn import venn3
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


## データセットをロード
dataset1=[]
dataset2a=[]
dataset2b=[]
dataset3=[]

for cv in [1,2,3]:
  list1  = np.load("data/gcng_input/exp_matrix/pattern1/" + str(cv) + '_test_gene_list.npy')
  list2a = np.load("data/gcng_input/exp_matrix/pattern2a/" + str(cv) + '_test_gene_list.npy')
  list3  = np.load("data/gcng_input/exp_matrix/pattern3/" + str(cv) + '_test_gene_list.npy')
  y1  = np.load("data/gcng_input/exp_matrix/pattern1/" + str(cv) + '_test_y.npy')
  y2a = np.load("data/gcng_input/exp_matrix/pattern2a/" + str(cv) + '_test_y.npy')
  y3  = np.load("data/gcng_input/exp_matrix/pattern3/" + str(cv) + '_test_y.npy')

  for i in range(len(list1)):
    if y1[i]==1:
      list1[i] = sorted(list1[i])
      dataset1.append(list1[i][0]+"-"+list1[i][1])
  for i in range(len(list2a)):
    if y2a[i]==1:
      list2a[i] = sorted(list2a[i])
      dataset2a.append(list2a[i][0]+"-"+list2a[i][1])
  for i in range(len(list3)):
    if y3[i]==1:
      list3[i] = sorted(list3[i])
      dataset3.append(list3[i][0]+"-"+list3[i][1])

for cv in [1,2]:
  temp2b = np.load("data/gcng_input/exp_matrix/pattern2b/" + str(cv) + '_test_gene_list.npy')
  y2b = np.load("data/gcng_input/exp_matrix/pattern2b/" + str(cv) + '_test_y.npy')

  for i in range(len(temp2b)):
    if y2b[i]==1:
      temp2b[i] = sorted(temp2b[i])
      dataset2b.append(temp2b[i][0]+"-"+temp2b[i][1])


print("dataset1  : " + str(len(dataset1)))
print("dataset2a : " + str(len(dataset2a)))
print("dataset2b : " + str(len(dataset2b)))
print("dataset3  : " + str(len(dataset3)))



## ベン図で関係性を図示
venn3([set(dataset1), set(dataset2a), set(dataset3)], set_labels=('dataset1', 'dataset2a', 'dataset3'))
plt.title("1-2a-3")
plt.savefig("data/gcng_input/exp_matrix/1-2a-3.pdf")
plt.clf()

venn3([set(dataset1), set(dataset2b), set(dataset3)], set_labels=('dataset1', 'dataset2b', 'dataset3'))
plt.title("1-2b-3")
plt.savefig("data/gcng_input/exp_matrix/1-2b-3.pdf")
plt.clf()

venn2([set(dataset2a), set(dataset2b)], set_labels=('dataset2a', 'dataset2b'))
plt.title("2a-2b")
plt.savefig("data/gcng_input/exp_matrix/2a-2b.pdf")
plt.clf()

print("~ 作成したベン図をチェック ~ ")



