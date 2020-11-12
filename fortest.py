#coding=utf-8
import numpy as np
from numpy import *
import pandas as pd
import time
import csv
import codecs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
  file_csv = codecs.open(file_name,'w+','utf-8')#追加
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("保存文件成功，处理结束")

bestFitList=list()

data_write_csv("F:/1 for pycharm/result/ionosphere/ionosphere.csv",bestFitList)
data_write_csv("F:/1 for pycharm/result/ionosphere/ionosphere.csv",bestAccList)
data_write_csv("F:/1 for pycharm/result/ionosphere/ionosphere.csv",time_end - time_start)
# def calcuAcc(whale, sample, label):
#     d = list(whale[:].nonzero())
#     data_X = sample[:, d[0]]
#     data_y = np.array(label)
#     # print(data_X)
#     # print(data_y)
#     kf = KFold(n_splits=10, shuffle=True)
#     i = 0
#     acc = list()
#     for train_index, test_index in kf.split(data_X):
#         i += 1
#         train_X, test_X = data_X[train_index], data_X[test_index]
#         train_y, test_y = data_y[train_index], data_y[test_index]
#         # knn
#         knn = KNeighborsClassifier()
#         knn.fit(train_X, train_y)
#         predict_y = knn.predict(test_X)
#         acc.append(balanced_accuracy_score(test_y, predict_y))
#     return (mean(acc))
#
# a=np.arange(225)
# b=a.reshape(15,15)
# whale1=np.zeros([15,15],dtype=int)
# whale2=np.ones([15,15],dtype=int)
# # sample=[[1,2,3],[4,5,6],[7,8,9]]
# # sample=np.array(sample)
# label=[1,0,1,0,0,0,1,0,0,1,0,1,0,1,1]
# t=calcuAcc(whale1[1],b,label)
# print(t)

def dataset():
    pass

# def calcuDistance(sample,label,num_whale,num_feat):
#     #计算异类距离与同类距离
#     db=0;dw=0
#     disdb=list();disdw=list()
#     d=0
#     for i in range(num_whale):
#         for j in range(num_whale):
#             if (label[i] == label[j]):
#                 for m in range(num_feat):
#                     if(sample[i,m]==sample[j,m]):
#                         d+=1
#                 disdw.append(d)
#                 d=0
#             else:
#                 for m in range(num_feat):
#                     if(sample[i,m]==sample[j,m]):
#                         d+=1
#                 disdb.append(d)
#                 d=0
#         dw+=max(disdw)/num_whale
#         db+=min(disdb)/num_whale
#     db/=num_feat
#     dw/=num_feat
#     return 1/(1+math.exp(-5*(db-dw)))
#
# a=[[1,0,1,0,1],[1,0,1,0,1],[2,0,2,0,2],[2,0,2,0,2]]
# a=np.array(a)
# label=[1,1,2,2]
# num1=4
# num2=5
# t=calcuDistance(sample=a,label=label,num_whale=4,num_feat=5)
# print(t)
# print(3/5)