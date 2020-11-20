# coding=utf-8
import numpy as np
from numpy import *
import pandas as pd
import time
import csv
import codecs
import xlwt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def calcuAcc(whale, sample, label):
    d = list(whale[:].nonzero())
    data_X = sample[:, d[0]]
    data_y = np.array(label)
    # print(data_X)
    # print(data_y)
    kf = KFold(n_splits=10, shuffle=True)
    i = 0
    acc = list()
    for train_index, test_index in kf.split(data_X):
        i += 1
        train_X, test_X = data_X[train_index], data_X[test_index]
        train_y, test_y = data_y[train_index], data_y[test_index]
        # knn
        knn = KNeighborsClassifier()
        knn.fit(train_X, train_y)
        predict_y = knn.predict(test_X)
        acc.append(balanced_accuracy_score(test_y, predict_y))
    return (mean(acc))


def calcuDistance(whale, sample, label, num_whale, num_feat):
    d = list(whale[:].nonzero())
    # data_X = sample[:, d[0]]
    # 计算异类距离与同类距离
    db = 0;
    dw = 0
    disdb = list();
    disdw = list()
    d = 0
    for i in range(num_whale):
        for j in range(num_whale):
            if (label[i] == label[j]):
                for m in range(d):
                    if (sample[i, m] == sample[j, m]):
                        d += 1
                disdw.append(d)
                d = 0
            else:
                for m in range(d):
                    if (sample[i, m] == sample[j, m]):
                        d += 1
                disdb.append(d)
                d = 0
        dw += max(disdw) / num_whale
        db += min(disdb) / num_whale
    db /= num_feat
    dw /= num_feat
    return 1 / (1 + math.exp(-5 * (db - dw)))

def basic_set(df):
    basic = {}
    for i in df.drop_duplicates().values.tolist():
        basic[str(i)]=[]
        for j,k in enumerate(df.values.tolist()):
            if k == i:
                basic[str(i)].append(j)
    return basic

def rough_set(whale, sample, label):
    label =pd.DataFrame(label)
    sample = pd.DataFrame(sample)
    data = pd.concat([sample,label],axis=1)
    # data = pd.DataFrame(data)
    data = data.dropna(axis=0,how='any')
    x_data = data.iloc[:,0:len(data.values[0]) -1]
    y_data = data.iloc[:,-1]
    # print(x_data)
    # print(y_data)
    #决策属性基本集
    y_basic_set = sorted([v for k,v in basic_set(y_data).items()])
    #条件属性基本集
    whale = np.array(whale)
    d = list(whale.nonzero())
    x_data_selected = x_data.iloc[:,d[0]]
    x_basic_set = sorted([v for k,v in basic_set(x_data_selected).items()])
    pos = []
    for i in x_basic_set:
        for j in y_basic_set:
            if set(i).issubset(j):
                pos.append(i)
    pos.sort()
    # print('pos:',pos)
    r_x_y = len([k for i in pos for k in i])/len(data)
    # print(r_x_y)
    return r_x_y

def calcuFitness(particle, i, sample, label, num_whale, num_feat,y):
    # y = 0.5
    # 计算依赖度和特征占比
    d = len(list(particle[i].nonzero()))
    fit = y * rough_set(particle[i],sample, label) \
          + (1 - y) * (num_feat-d)/num_feat
    # fit = y * calcuAcc(particle[i], sample, label) \
    #       + (1 - y) * calcuDistance(particle[i], sample, label, num_whale, num_feat)
    acc = calcuAcc(particle[i], sample, label)
    return fit, acc;


def calcuD(list1, list2, c):
    """

    :param list1: rand or best
    :param list2: particle i
    :return: 距离
    """
    m = len(list1)
    d1 = 0;
    d2 = 0;
    d3 = 0
    for i in range(m):
        if (list1[i] == list2[i]):
            d1 += 1
        if (list1[i] == 1):
            d2 += 1
        if (list2[i] == 1):
            d3 += 1
    if ((c + 1) * m - d2 * c - d3 == 0):
        return 1
    return d1 / ((c + 1) * m - d2 * c - d3)


def woa(maxt,y):
    time_start = time.time()
    print("程序运行")
    fh.write('\n' + '程序运行' )
    df = pd.read_csv('F:/1 for pycharm/dataset/ionosphere.csv', header=None)
    data = df.values[:, 0:len(df.values[0]) - 1]
    scaler = StandardScaler()
    sample = scaler.fit_transform(data)
    # ionosphere.csv数据集第二列全为零
    sample[:, 1] = 0
    # 标签
    label = df.values[:, -1]
    cla = list()
    for i in label:
        if (i == 'g'):
            cla.append('1')
        else:
            cla.append('0')
    print("粒子初始化")
    fh.write('\n' + '粒子初始化' )
    # 初始化量子蓝鲸
    if (df.shape[0] / 20 < 20):
        num_whale = 20
    else:
        num_whale = df.shape[0] / 20
    num_feat = df.shape[1] - 1
    # num_whale = 20;
    # num_feat = 34
    particlea = np.random.rand(num_whale, num_feat)
    # particleb=np.random.rand(num_whale,num_feat)
    particlea[:, :] = math.pi / 4
    # particlea[:,:]=1/math.sqrt(2)
    # particleb[:,:]=1/math.sqrt(2)
    particle2 = np.empty([num_whale, num_feat])
    for x in range(num_whale):
        for y in range(num_feat):
            # 观测
            r = random.random()
            if (r > math.pow(math.sin(particlea[x][y]), 2)):
                particle2[x][y] = 1
                # 列表fitness存放每个粒子的适应度（准确率加距离）
                # flag = 0
            else:
                particle2[x][y] = 0
                # 当初始化全为零时，KNN无法fit,随机选一个作为最优,
                # 且给一个标记，因为在搜寻阶段还要KNN来fit
                # flag = 1
        # if (flag == 0):
    fitness = list()
    accuracy = list()
    for i in range(num_whale):
        fit, acc = calcuFitness(particle2, i, sample, cla, num_whale, num_feat,y)
        fitness.append(fit)
        accuracy.append(acc)
    bestFitList = list()
    bestAccList = list()
    bestFitness = max(fitness)
    bestIndex = fitness.index(bestFitness)
    best = particle2[bestIndex]
    # else:
    #     # 当初始化全为零时，KNN无法fit,随机选一个作为最优,
    #     # 且给一个标记，因为在搜寻阶段还要KNN来fit
    #     fitness = list()
    #     bestFitList = list()
    #     randbest = random.randint(0, num_whale)
    #     bestFitness = 0
    #     best = particle2[randbest]
    # 迭代次数t,最大迭代次数maxT=20
    t = 0;
    # maxt = 20
    # maxt = 200
    print("进入迭代")
    fh.write('\n' + '进入迭代' )
    while (t < maxt):
        # print("####t####:", t)

        t += 1
        for x in range(num_whale):
            for y in range(num_feat):
                # 量子观测
                r = random.random()
                if (r > math.pow(math.sin(particlea[x][y]), 2)):
                    particle2[x][y] = 1
                else:
                    particle2[x][y] = 0
        for i in range(num_whale):
            # 公式里的参数r
            r1 = random.random()
            # 参数a:A=2*a1*r-a1
            a = 2 * (2 * r1 - 1) * (maxt - t) / maxt
            # 参数p，用来选择更新方案
            p = random.random()
            if (p >= 0.5):
                # flag = 0
                # 螺旋
                for j in range(num_feat):
                    # 公式里的参数r
                    r1 = random.random()
                    # 参数a:A=2*a1*r-a1
                    a = 2 * (2 * r1 - 1) * (maxt - t) / maxt
                    d = calcuD(list1=best, list2=particle2[i], c=2 * r1)
                    particlea[i, j] = particlea[i, j] - a * d
            elif (math.fabs(a) < 1):
                # 公式里的参数r
                r1 = random.random()
                # 参数a:A=2*a1*r-a1
                a = 2 * (2 * r1 - 1) * (maxt - t) / maxt
                d = calcuD(list1=best, list2=particle2[i], c=2 * r1)
                for j in range(num_feat):
                    particlea[i, j] = best[j] - a * d
            else:
                # 搜寻猎物
                # if (flag == 0):
                # print("flag", flag)
                # flag = 0
                rand1 = random.randint(0, num_whale)
                rand2 = random.randint(0, num_whale)
                fit1 = calcuFitness(particle2, i, sample, cla, num_whale, num_feat,y)
                fit2 = calcuFitness(particle2, i, sample, cla, num_whale, num_feat,y)
                p = 0.75  # 竞赛参数
                rjingsai = random.random()
                if (fit1 > fit2):
                    if (rjingsai < p):
                        for j in range(num_feat):
                            # 公式里的参数r
                            r1 = random.random()
                            # 参数a:A=2*a1*r-a1
                            a = 2 * (2 * r1 - 1) * (maxt - t) / maxt
                            d = calcuD(list1=particle2[rand1], list2=particle2[i], c=2 * r1)
                            particlea[i, j] = particlea[rand1, j] - a * d
                    else:
                        for j in range(num_feat):
                            # 公式里的参数r
                            r1 = random.random()
                            # 参数a:A=2*a1*r-a1
                            a = 2 * (2 * r1 - 1) * (maxt - t) / maxt
                            d = calcuD(list1=particle2[rand1], list2=particle2[i], c=2 * r1)
                            particlea[i, j] = particlea[rand2, j] - a * d
                else:
                    if (rjingsai < p):
                        for j in range(num_feat):
                            # 公式里的参数r
                            r1 = random.random()
                            # 参数a:A=2*a1*r-a1
                            a = 2 * (2 * r1 - 1) * (maxt - t) / maxt
                            d = calcuD(list1=particle2[rand1], list2=particle2[i], c=2 * r1)
                            particlea[i, j] = particlea[rand2, j] - a * d
                    else:
                        for j in range(num_feat):
                            # 公式里的参数r
                            r1 = random.random()
                            # 参数a:A=2*a1*r-a1
                            a = 2 * (2 * r1 - 1) * (maxt - t) / maxt
                            d = calcuD(list1=particle2[rand1], list2=particle2[i], c=2 * r1)
                            particlea[i, j] = particlea[rand1, j] - a * d
            # else:
            #     flag = 0
            #     flagrand = random.randint(0, num_whale)
            #     d = calcuD(list1=particle2[flagrand], list2=particle2[i], c=2 * r1)
            #     for j in range(num_feat):
            #         particlea[i, j] = particlea[flagrand, j] - a * d
        # if (flag == 0):
        # print("flag",flag)
        # 求最优蓝鲸个体
        fitness = list()
        accuracy = list()
        for i in range(num_whale):
            fit, acc = calcuFitness(particle2, i, sample, cla, num_whale, num_feat,y)
            fitness.append(fit)
            accuracy.append(acc)
        maxFitness = max(fitness)
        maxIndex = fitness.index(maxFitness)
        bestFitList.append(maxFitness)
        maxAccuracy = accuracy[maxIndex]
        bestAccList.append(maxAccuracy)
        if (maxFitness > bestFitness):
            bestFitness = maxFitness
            bestIndex = maxIndex
            best = particle2[bestIndex]
            # 将particlea的值规范化到规划,由于是角度似乎又不用
        # else:
        #     randbest = random.randint(0, num_whale)
        #     bestFitness = 0
        #     best = particle2[randbest]
    for y in range(num_feat):
        # 量子观测
        r2 = random.random()
        if (r2 > math.pow(math.sin(best[y]), 2)):
            particle2[x][y] = 1
        else:
            particle2[x][y] = 0
    x = list(range(maxt))
    plt.plot(x, bestFitList, label="fit")
    plt.plot(x, bestAccList, label="acc")
    plt.show()
    m = 0
    for i in range(num_feat):
        if (best[i] > 0):
            m += 1
    print('最优秀蓝鲸：',best)
    fh.write('\n' + '####最优秀蓝鲸：=' + str(best) + "#####")
    # print('len(best)', len(best))
    print('特征选择个数', m)
    fh.write('\n' + '####特征选择个数：=' + str(m) + "#####")
    print('最优适应度',bestFitness)
    fh.write('\n' + '####最优适应度：=' + str(bestFitness) + "#####")
    bestaccuracy=max(bestAccList)
    print("最优分类精度：",bestaccuracy)
    fh.write('\n' + '####最优分类精度：=' + str(bestaccuracy) + "#####")
    # print(bestAccList)
    # print('bestFitnesslist', bestFitList)
    # print('bestacclist', bestAccList)
    time_end = time.time()
    # dataframe = pd.DataFrame(bestFitList)
    # dataframe.to_excel('F:/1 for pycharm/result/ionosphere/ionospherefitness1.xls')
    # dataframe = pd.DataFrame(bestAccList)
    # dataframe.to_excel('F:/1 for pycharm/result/ionosphere/ionosphereaccuracy1.xls')
    print('totally cost', time_end - time_start)
    # fh.write('\n' + 'totally cost：' + time_end - time_start)
    return bestFitness

betterfit=0
bettert=0
bettery=0
for t in [5,10,20,30,40,50,60,70,80,90,100,150]:
    for y in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print("####迭代变量=",t,"#####")
        fh = open('F:/1 for pycharm/result/ionosphere/ionospherediedai.txt', 'a+', encoding='utf-8')
        fh.write('\n'+'####迭代变量='+str(t)+"#####")
        print("####参数y=",y,"#####")
        # fh = open('F:/1 for pycharm/result/ionosphere/ionospherediedai.txt', 'a+', encoding='utf-8')
        fh.write('\n'+'####参数y='+str(y)+"#####")
        fit1 =woa(t,y)
        if (betterfit<fit1):
            betterfit=fit1
            bettert = t
            bettery = y
print(betterfit)
fh = open('F:/1 for pycharm/result/ionosphere/ionospherediedai.txt', 'a+', encoding='utf-8')
fh.write('betterfit'+betterfit+'\n')
print(bettert)
# fh = open('F:/1 for pycharm/result/ionosphere/ionospherediedai.txt', 'a+', encoding='utf-8')
fh.write('bettert:'+bettert+'\n')
# fh = open('F:/1 for pycharm/result/ionosphere/ionospherediedai.txt', 'a+', encoding='utf-8')
# fh.write(bettery)
fh.write('bettert:'+bettery+'\n')
print(bettery)