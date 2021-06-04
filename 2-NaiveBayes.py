import numpy as np
import time


'''
数据集：Mnist
------------------------------
运行结果：
    正确率：84.3%  运行时长：50s
'''

def load_data(file_path):
    x = []
    y = []

    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        x.append([int(int(i) > 128) for i in line[1:]])  # 将每个特征值二值化处理为0和1
        y.append(int(line[0]))
    return x, y


def NaiveBayes(Py, Px_y, x):
    '''
    通过朴素贝叶斯进行概率估计
    Py: 先验概率分布
    Px_y: 条件概率分布
    x: 要估计的样本x
    返回所有label的估计概率
    '''
    feature_num = 784  # 特征数
    class_num = 10     # 类别数

    P = [0] * class_num
    # 对每个类别单独估计概率
    for i in range(class_num):
        #在训练过程中对概率进行了log处理，这里连乘所有概率变加，最后比较哪个类别概率最大
        sum = 0
        # 获得每个特征的概率
        for j in range(feature_num):
            sum += Px_y[i][j][x[j]]
        # 条件概率 + 先验概率
        P[i] = sum + Py[i]
    # 找到最大概率返回起类别
    return P.index(max(P))


def Evaluate(Py, Px_y, test_x, test_y):
    error = 0
    for i in range(len(test_x)):
        print('测试集中第%d/%d组样本' % (i, len(train_x)))
        x = test_x[i]
        pred_y = NaiveBayes(Py, Px_y, x)

        if pred_y != test_y[i]:
            error += 1
    return (1 - error / len(test_x))


# 计算先验概率Py和条件概率Px_y=P（X=x|Y = y）
def get_Probability(train_x, train_y):
    feature_num = 784
    class_num = 10
    Py = np.zeros((class_num * 1))
    Px_y = np.zeros((class_num, feature_num, 2))  # 每个特征两种取值

    # 计算先验概率, 避免出现0的情况分子加1分母加类别数
    for i in range(class_num):
        Py[i] = (np.sum(np.mat(train_y) == i) + 1) / (len(train_y) + 10)
    Py = np.log(Py)

    # 先统计训练集样本每个特征出现的次数
    for i in range(len(train_x)):
        x = train_x[i]
        label = train_y[i]
        for j in range(feature_num):
            Px_y[label][j][x[j]] += 1
    # 计算条件概率
    for label in range(class_num):
        for j in range(feature_num):
            # 计算对于y=label，x第j个特征为0和1的条件概率分布
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))      

    return Py, Px_y



if __name__ == '__main__':
    train_x, train_y = load_data('./Mnist/mnist_train.csv')
    test_x, test_y = load_data('./Mnist/mnist_test.csv')

    # 开始训练，学习先验概率分布和条件概率分布
    Py, Px_y = get_Probability(train_x, train_y)
    start = time.time()
    # 开始测试，用学习到的先验概率分布和条件概率分布对测试集进行测试
    acc = Evaluate(Py, Px_y, test_x, test_y)
    end = time.time()

    print('贝叶斯模型的精度: %d' % (acc * 100), '%')
    print('时间跨度: %d' % (end - start))

