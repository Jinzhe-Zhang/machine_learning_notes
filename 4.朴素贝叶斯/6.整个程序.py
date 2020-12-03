import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from collections import Counter


# 正态分布（高斯分布）函数，输入值x，miu均值，rou方差
def Gaussian(x, miu, rou):
    p_xc = 1 / (np.sqrt(2 * np.pi) * rou) * np.exp(-(x - miu) ** 2 / (2 * rou ** 2))
    return p_xc


# 朴素贝叶斯算法 极大似然估计，计算结果与《统计学习方法》李航，P51页一致
def get_p_matrix(arr_data):
    # 获得y中分类标签的唯一值
    y_lables = np.unique(arr_data[:, -1])
    # y_lables = set(arr_data[:,-1])  # 同上，两种写法均可

    y_counts = len(arr_data)  # y总数据条数
    y_p = {}  # y中每一个分类的概率，字典初始化为空，y分类数是不定的，按字典存储更方便取值
    for y_lable in y_lables:
        y_p[y_lable] = len(arr_data[arr_data[:, -1] == y_lable]) / y_counts  # y中每一个分类的概率（其实就是频率）

    yx_cnt = []  # 固定y以后的，x中每一种特征出现的次数，此数据量并不大，y分类数*x维度列数，按list存储即可
    for y_lable in y_p.keys():  # 先固定y，遍历y中每一个分类
        y_lable_cnt = len(arr_data[arr_data[:, -1] == y_lable])  # 此y分类数据条数
        for x_j in range(0, arr_data.shape[1] - 1):  # 在固定x特征列，遍历每列x中的特征
            x_j_count = Counter(
                arr_data[arr_data[:, -1] == y_lable][:, x_j])  # 按列统计每种特征出现的次数，因为某一列的特征数是不固定的，所以按dict类型存储
            yx_cnt.append([y_lable, y_lable_cnt, x_j, dict(x_j_count)])

    yx_p = []  # 将统计次数处理为概率
    for i in range(0, len(yx_cnt)):
        # print(yx_cnt[i])
        # print(yx_cnt[i][3])
        p = {}  # 将每列x特征出现的次数转换为概率
        for key in yx_cnt[i][3].keys():
            p[key] = yx_cnt[i][3][key] / yx_cnt[i][1]
        yx_p.append([yx_cnt[i][0], yx_cnt[i][1], yx_cnt[i][2], p])
    return y_p, yx_p


# 朴素贝叶斯算法   贝叶斯估计， λ=1  K=2， S=3； λ=1 拉普拉斯平滑，计算结果与《统计学习方法》李航，P52页一致。
def get_p_matrix_laplace(arr_data):
    # 获得y中分类标签的唯一值
    y_lables = np.unique(arr_data[:, -1])
    # y_lables = set(arr_data[:,-1])  # 同上，两种写法均可
    lambda1 = 1  # λ=1 拉普拉斯平滑
    k = len(y_lables)  # y分类个数k，用于拉普拉斯平滑

    y_counts = len(arr_data)  # y总数据条数
    y_p = {}  # y中每一个分类的概率，字典初始化为空，y分类数是不定的，按字典存储更方便取值
    for y_lable in y_lables:
        y_p[y_lable] = (len(arr_data[arr_data[:, -1] == y_lable]) + lambda1) / (
                    y_counts + k * lambda1)  # y中每一个分类的概率（其实就是频率）

    yx_cnt = []  # 固定y以后的，x中每一种特征出现的次数，此数据量并不大，y分类数*x维度列数，按list存储即可
    for y_lable in y_p.keys():  # 先固定y，遍历y中每一个分类
        y_lable_cnt = len(arr_data[arr_data[:, -1] == y_lable])  # 此y分类数据条数,N
        for x_j in range(0, arr_data.shape[1] - 1):  # 在固定x特征列，遍历每列x中的特征
            x_j_count = Counter(
                arr_data[arr_data[:, -1] == y_lable][:, x_j])  # 按列统计每种特征出现的次数，因为某一列的特征数是不固定的，所以按dict类型存储
            yx_cnt.append([y_lable, y_lable_cnt, x_j, dict(x_j_count)])

    yx_p = []  # 将统计次数处理为概率
    for i in range(0, len(yx_cnt)):
        # print(yx_cnt[i])
        # print(yx_cnt[i][3])
        p = {}  # 将每列x特征出现的次数转换为概率
        s = len(yx_cnt[i][3].keys())
        for key in yx_cnt[i][3].keys():
            p[key] = (yx_cnt[i][3][key] + lambda1) / (yx_cnt[i][1] + s * lambda1)
        yx_p.append([yx_cnt[i][0], yx_cnt[i][1], yx_cnt[i][2], p])
    return y_p, yx_p


if __name__ == "__main__":
    df_data = pd.read_csv('./naivebayes_data.csv')
    arr_data = np.array(df_data.values)  # 数据处理为numpy.array类型，其实pandas.Dataframe类型更方便计算
    # 测试数据一条，x1,x2
    features = [2, 'S']

    # 1、素贝叶斯算法，极大似然估计，调用函数，计算概率矩阵
    y_p, yx_p = get_p_matrix(arr_data)
    # 查看概率矩阵
    print('1、素贝叶斯算法，极大似然估计：')
    print('y_p:\n', y_p)
    print('yx_p:\n', yx_p)

    # 朴素贝叶斯分类器，数据[2,'S']手动计算过程
    # P(c|x)=P(c)P(x|c)/P(x)，同一组数据，对所有分类来说分母相同，所已只比较分子大小即可
    # c=1， 0.6*0.3333*0.1111=0.022217778
    # c=-1，0.4*0.3333*0.5=0.06666
    # 比较c=1及c=-1时概率大小，数据[2,'S']数据c=-1类

    yx_p_arr = np.array(yx_p)  # list类型不好按列取值，转换为array类型
    # 编程计算每个分类的概率值
    res1 = {}
    for key in y_p.keys():
        res1[key] = 1 * y_p[key]
        for i in range(0, len(features)):
            res1[key] = res1[key] * yx_p_arr[(yx_p_arr[:, 0] == key) & (yx_p_arr[:, 2] == i)][:, 3][0][features[i]]

    print('测试数据:', features, '各分类概率：', res1, ',预测结果为：', max(res1, key=res1.get))

    # 2、朴素贝叶斯算法，贝叶斯估计，调用函数，计算概率矩阵
    y_p_ll, yx_p_ll = get_p_matrix_laplace(arr_data)
    print('\n 2、朴素贝叶斯算法,贝叶斯估计')
    print('y_p_ll:\n', y_p_ll)
    print('yx_p_ll:\n', yx_p_ll)

    yx_p_ll_arr = np.array(yx_p_ll)  # list类型不好按列取值，转换为array类型
    # 编程计算每个分类的概率值
    res2 = {}
    for key in y_p_ll.keys():
        res2[key] = 1 * y_p_ll[key]
        for i in range(0, len(features)):
            res2[key] = res2[key] * yx_p_ll_arr[(yx_p_ll_arr[:, 0] == key) & (yx_p_ll_arr[:, 2] == i)][:, 3][0][
                features[i]]

    print('测试数据:', features, '各分类概率：', res2, ',预测结果为：', max(res2, key=res2.get))

