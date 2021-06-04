import numpy as np
import time

'''
数据集：Mnist, 特征维度28*28=784， 标签类别为0-9
训练集数量：60000 测试集数量：10000（实际使用：200）
------------------------------
运行结果：（邻近k数量：25）
向量距离使用算法——欧式距离
    正确率：97%  运行时长：302s
向量距离使用算法——曼哈顿距离
    正确率：14%  运行时长：246s
'''

def load_data(file_path):
    x = []
    y = []

    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        x.append([int(i) for i in line[1:]])
        y.append(int(line[0]))
    return x, y


def compute_distance(x1, x2):
    '''计算欧式距离'''
    return np.sqrt(np.sum(np.square(x1 - x2)))
    # 曼哈顿距离计算公式
    # return np.sum(x1 - x2)


def KNN(train_x, train_y, x, K):
    
    dis_all = [0] * len(train_x)

    for i in range(len(train_x)):
        train_i = train_x[i]
        dis = compute_distance(train_i, x)
        dis_all[i] = dis
    topk_index = np.argsort(np.array(dis_all))[:K]  # 升序
    count_label = [0] * 10  # 标签有10个类别
    for index in topk_index:
        count_label[int(train_y[index])] += 1
    return count_label.index(max(count_label))



def Evaluate(train_x, train_y, test_x, test_y, K=25):
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).T
    test_x = np.mat(test_x)
    test_y = np.mat(test_y).T
   
    error_count = 0
    for i in range(len(test_x)):
        print('test %d/%d' % (i, len(test_x)))
        x_i = test_x[i]
        y_i = test_y[i]
        pred_y = KNN(train_x, train_y, x_i, K)

        if pred_y != y_i:
            error_count += 1
    return 1 - (error_count / len(test_x))



if __name__ == '__main__':
    start = time.time()
    train_x, train_y = load_data('./Mnist/mnist_train.csv')
    test_x, test_y = load_data('./mnist/mnist_test.csv')

    K = 25
    acc = Evaluate(train_x, train_y, test_x, test_y, K)
    print('test在knn模型上的精度: %d' % (acc * 100), '%')
    
    end = time.time()
    print('测试时间:', end - start)

