'''
感知机： y = sign(w * x + b), 是一个二分类模型
当 -yi * (w * xi + b) >= 0 判断为误分类点
'''

import numpy as np
import time


def load_data(file_path):
    x = []
    y = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            x.append([int(i) / 255 for i in line[1:]])
            if int(line[0]) >= 5:
                y.append(1)
            else:
                y.append(-1)
    return x, y


def perceptron(train_x, train_y, epochs):
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).T 
    m, n = np.shape(train_x)  # m, n分别代表样本数和特征数
    
    w = np.zeros((1, n))
    b = 0
    lr = 0.0001

    for epoch in range(epochs):
        for i in range(m):
            xi = train_x[i]
            yi = train_y[i]

            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + lr * yi * xi
                b = b + lr * yi
        # print(w, b)
    return w, b 


def Evaluate(test_x, test_y, w, b):
    test_x = np.mat(test_x)
    test_y = np.mat(test_y).T

    m, n = np.shape(test_x)
    error = 0

    for i in range(m):
        print('test %d/%d sample' % (i, len(test_y)))
        xi = test_x[i]
        yi = test_y[i]
        
        if -1 * yi *(w*xi.T + b) >= 0:
            error += 1
    return (1 - error/m)


if __name__ == '__main__':
    start = time.time()
    train_x, train_y = load_data('./Mnist/mnist_train.csv')
    test_x, test_y = load_data('./Mnist/mnist_test.csv')

    w, b = perceptron(train_x, train_y, epochs=50)
    print(w.shape, b)

    acc = Evaluate(test_x, test_y, w, b)
    end = time.time()
    print('test acc is: %d' % (acc * 100 ), '%')
    print('time is %d' % (end - start))
    