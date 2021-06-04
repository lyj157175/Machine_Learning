'''
数据集：Mnist
    正确率：98.91%
    运行时长：59s
'''

import time
import numpy as np


def loadData(fileName):
    dataList = []
    labelList = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)
        dataList.append([int(num)/255 for num in curLine[1:]])
    return dataList, labelList



def predict(w, x):
    pass


def logisticRegression(trainDataList, trainLabelList, iter = 200):
    pass 


def model_test(testDataList, testLabelList, w):
    #与训练过程一致，先将所有的样本添加一维，值为1，理由请查看训练函数
    for i in range(len(testDataList)):
        testDataList[i].append(1)

    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(w, testDataList[i]):
            errorCnt += 1
    return 1 - errorCnt / len(testDataList)



if __name__ == '__main__':
    start = time.time()
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    # 开始训练，学习w
    w = logisticRegression(trainData, trainLabel)

    accuracy = model_test(testData, testLabel, w)
    print('the accuracy is:', accuracy)
    print('time span:', time.time() - start)

