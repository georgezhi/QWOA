# _*_coding:utf-8 _*_
#@Time    :2018/12/22 上午10:45
#@Author  :we2swing
#@FileName: test.py

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def basic_set(df):
    '''
    #决策域划分
    :param df:
    :return:
    '''
    basic = {}
    for i in df.drop_duplicates().values.tolist():
        basic[str(i)] = []
        for j, k in enumerate(df.values.tolist()):
            if k == i:
                basic[str(i)].append(j)
    return basic

def Euclidean(x_data):
    n = x_data.shape[1]#代表列数
    m = x_data.shape[0]#代表行数
    num = [0] * n
    num2 = np.zeros((m,m))
    #转为list
    arr = np.array(x_data)

    counti,countj,countk,countz = 0,0,0,0
    #遍历每一行的元素
    for j in arr:
        countk = 0
        for k in arr:
            if countj != countk:
                countz = 0
                # 遍历每一行的每一个元素
                for z in k:
                    temp = np.square(z - j[countz])
                    if countz == 0:
                        num[countz] = temp
                    else:
                        #把每一行中每个元素的差值平方相加
                        num[countz] = num[countz-1] + temp
                    countz = countz + 1
            else:
                num[countz-1] = 0
            num2[countj][countk] = round(math.sqrt(num[countz-1]),1)
            countk = countk + 1
        countj = countj + 1
    return num2

def key_basic(num,k):
    # k = 4  # 设δ=k
    # k = [4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0]
    num1 = {}
    counti, countj = 0, 0
    for i in num:
        num1[counti] = []
        countj = 0
        for j in i:
            if j <= k:
                num1[counti].append(countj)
            countj = countj + 1
        counti = counti + 1
    return num1

def compare(data):
    # k = [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
    global t1
    t1 = 4.0
    data = data.dropna(axis=0, how='any')
    x_data = data.drop(['judge'], axis=1)
    y_data = data.loc[:, 'judge']
    n = x_data.shape[1]  # 代表列数
    m = x_data.shape[0]  # 代表行数

    # 决策属性基本集
    y_basic_set = sorted([v for k, v in basic_set(y_data).items()])
    num = Euclidean(x_data)
    # print(num)
    x_basic_set = [v for k, v in key_basic(num,t1).items()]

    # γC(D)
    pos = []
    for i in x_basic_set:
        for j in y_basic_set:
            if set(i).issubset(j):
                pos.append(i)
    # pos.sort()#排序
    r_x_y = len(pos) / len(data)  # 也许可以写一个card函数
    print('依赖度r_x_(y):', r_x_y)

    # 计算每一个γ(C-a)(D)
    # 获取条件属性个数为总下标存入集合
    # 收集属性重要性
    imp_attr = []
    columns_num = list(range(len(x_data.columns)))
    for i in columns_num:
        c = columns_num.copy()
        c.remove(i)
        u = data.iloc[:, c]
        num = Euclidean(u)
        u = sorted([v for k, v in key_basic(num,t1).items()])

        # γC(D)
        pos_va = []
        for k in u:
            for j in y_basic_set:
                if set(k).issubset(j):
                    pos_va.append(k)
        r_x_y_2 = len(pos_va) / len(data)
        r_diff = round((r_x_y - r_x_y_2), 4)
        imp_attr.append(r_diff)

    dict_imp = {}
    for o, p in enumerate(imp_attr):
        dict_imp[data.columns[o]] = p

    result = dict_imp
    # print(imp_attr)
    return result

def sepdata(data):
    # 获取数据长度
    len = data.iloc[:, 0].size
    # 将数据划分
    if len % 100 != 0:
        if len > 100:
            num = len // 100 + 1
        else:
            num = 1
    else:
        if len > 100:
            num = int(len / 100)
        else:
            num = 1
    arr = [[]] * num

    count = 0
    for i in arr:
        # 如果数少于100或者最后一部分数少于100，则放入一个由数长决定的数组
        if num == 1:
            arr[count] = data.iloc[0:len]  # 取100开始，取
        elif count == num - 1:
            arr[count] = data.iloc[100 * count:len]
        else:
            arr[count] = data.iloc[100 * count:(count + 1) * 100]
        count = count + 1
    sorted_dict_imp = [[]] * num
    total = [0] * 27
    title = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
             'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27']
    # total = [0] * 16
    # title = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']
    print(sorted_dict_imp)
    count = 0
    for i in arr:
        print('-------------------------------------第%d个数据集数据-----------------------------------------' % (count + 1))
        sorted_dict_imp[count] = compare(i)
        count = count + 1
    count1 = 0
    # 将dict的key为C1-Cn的value存入total中保存,并且相加
    for i in sorted_dict_imp:
        count = 0
        if count1 == 0:
            for j in title:
                total[count] = i.get(j)
                count = count + 1
        else:
            for z in title:
                total[count] = i.get(z) + total[count]
                count = count + 1
        count1 = count1 + 1
    # 输出最终C1-Cn的结果
    count = 0
    for i in title:
        print(i, ':', round(total[count], 4))
        count = count + 1
    figure(total)

def figure(opt):
    count = 0
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    title = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
             'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27']
    plt.scatter(title, opt, c='b', s=7)
    plt.plot(title, opt, linewidth=3, alpha=0.7)
    plt.show()

def main():
    data = pd.read_csv(filepath_or_buffer='E:/dataset/test.csv')
    sepdata(data)

if __name__ == '__main__':
    main()