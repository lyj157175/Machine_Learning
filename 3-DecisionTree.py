import time 
import numpy as np


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


# 找到标签集中数量最多的那一类
def max_class(label_list):
    labels = {}
    for i in label_list:
        if i in labels:
            labels[i] += 1
        else:
            labels[i] = 1
    labels_sort = sorted(labels.items(), lambda x: x[1], reverse=True)
    return labels_sort[0][0]


def dataset_split(train_x, train_y, feature, a):
    train_x_sub = []
    train_y_sub = []

    for i in range(len(train_x)):
        if train_x[i][feature] == a:
            # 去掉特征feature, 当前特征值为a时保留当前行样本
            train_x_sub.append(train_x[i][:feature] + train_x[i][feature+1:])
            train_y_sub.append(train_y[i])
    # 返回更新的数据集
    return train_x_sub, train_y_sub


# 计算标签集的经验熵H(D) = - sum(ci/di * log(ci/di))
def get_H_D(train_y):
    H_D = 0  # 初始化经验熵
    train_y_set = set([i for i in train_y])
    for label in train_y_set:
        p = train_y[train_y == label].size / train_y.size
    H_D += -1 * p * np.log2(p)
    return H_D


def get_H_D_A(train_x_feature, train_y):
    '''
    计算当前特征数据的经验条件熵
    '''
    H_D_A = 0  # 初始化经验条件熵
    train_x_feature_set = set([i for i in train_x_feature])
    for i in train_x_feature_set:
        # H(D|A) = sum(|Di|/|D| * H(Di))
        H_D_A += train_x_feature[train_x_feature==i].size / train_x_feature.size \
            * get_H_D(train_y[train_x_feature==i])
    return H_D_A


def get_best_feature(train_x, train_y):
    '''
    计算信息增益最大的特征及最大信息增益值
    计算信息增益  g(D, A) = H(D) - H(D | A)
    '''
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    feature_num = train_x.shape[1]

    max_feature = -1  # 初始化信息增益最大特征
    max_g_d_a = -1    # 初始化最大信息增益

    h_d = get_H_D(train_y)  # 就算标签集的经验熵
    # 计算每个特征的经验条件熵
    for feature in range(feature_num):
        train_x_feature = np.array(train_x[:, feature].flat)
        h_d_a = get_H_D_A(train_x_feature, train_y)
        g_d_a = h_d - h_d_a
        
        if g_d_a > max_g_d_a:
            max_g_d_a = g_d_a
            max_feature = feature
    return max_feature, max_g_d_a
        

def create_tree(*dataset):
    Epsilon = 0.1  # 信息增益阈值
    train_x = dataset[0][0]
    train_y = dataset[0][1]
    # print('开始构建决策树，有%d个特征和%d个标签' % (len(train_x[0], len(train_y))))
    label = {i for i in train_y}  # 标签去重
    # 如果标签列表中所有标签值都一样，则不用在划分，返回当前标签值作为该叶子节点的值
    if len(label) == 0:
        return train_y[0]
    # 如果没有特征来进行划分，则返回当前标签列表中标签数最多的一类标签
    if len(train_x[0]) == 0:
        return max_class(train_y)
    
    # 计算信息增益和最大信息增益时的特征
    feature, max_gda = get_best_feature(train_x, train_y)
    if max_gda < Epsilon:
        return max_class(train_y)
    
    treedict = {feature:{}}
    # 分别进入特征值为0和1的分支
    treedict[feature][0] = create_tree(dataset_split(train_x, train_y, feature, 0))
    treedict[feature][1] = create_tree(dataset_split(train_x, train_y, feature, 1))
    return treedict
    


def predict(test_x, tree):
    while True:
        #因为有时候当前字典只有一个节点
        #例如{73: {0: {74:6}}}看起来节点很多，但是对于字典的最顶层来说，只有73一个key，其余都是value
        #若还是采用for来读取的话不太合适，所以使用下行这种方式读取key和value
        (key, value), = tree.items()
        #如果当前的value是字典，说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':
            #获取目前所在节点的feature值，需要在样本中删除该feature
            #因为在创建树的过程中，feature的索引值永远是对于当时剩余的feature来设置的
            #所以需要不断地删除已经用掉的特征，保证索引相对位置的一致性
            dataVal = test_x[key]
            del test_x[key]
            #将tree更新为其子节点的字典
            tree = value[dataVal]
            #如果当前节点的子节点的值是int，就直接返回该int值
            #例如{403: {0: 7, 1: {297:7}}，dataVal=0
            #此时上一行tree = value[dataVal]，将tree定位到了7，而7不再是一个字典了，
            #这里就可以直接返回7了，如果tree = value[1]，那就是一个新的子节点，需要继续遍历下去
            if type(tree).__name__ == 'int':
                #返回该节点值，也就是分类值
                return tree
        else:
            #如果当前value不是字典，那就返回分类值
            return value


def Evaluate(test_x, test_y, tree):
    error = 0
    for i in range(len(test_y)):
        print('测试第%d/%d条数据' % (i, len(test_y)))
        pred_y = predict(test_x[i], tree)
        if pred_y != test_y[i]:
            error += 1
    return (1 - error/len(test_x))
        


if __name__ == '__main__':
    train_x, train_y = load_data('./Mnist/mnist_train.csv')
    test_x, test_y = load_data('./Mnist/mnist_test.csv')

    # 开始训练，学习先验概率分布和条件概率分布
    tree = create_tree((train_x, train_y))
    start = time.time()
    # 开始测试，用学习到的先验概率分布和条件概率分布对测试集进行测试
    acc = Evaluate(test_x, test_y, tree)
    end = time.time()

    print('贝叶斯模型的精度: %d' % (acc * 100), '%')
    print('时间跨度: %d' % (end - start))