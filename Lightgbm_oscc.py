import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, \
    KMeans
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import model_selection as cv, tree
from sklearn import metrics,preprocessing
# Import some data to play with
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef, f1_score, accuracy_score, precision_score
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, global_mean_pool, global_max_pool, global_add_pool

import torch.nn.functional as F
from torch.nn import Parameter

import warnings
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, \
    KMeans
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from deepforest import CascadeForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import model_selection as cv, tree
from sklearn import metrics,preprocessing
# Import some data to play with
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef, f1_score, accuracy_score, precision_score
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, global_mean_pool, global_max_pool, global_add_pool

import torch.nn.functional as F
from torch.nn import Parameter

import warnings
import pydotplus
import graphviz
warnings.filterwarnings('ignore')
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
import psutil
import os
import torch
import warnings
def dataread(fname):
    s = np.loadtxt(fname, dtype=np.float32, delimiter=' ')
    qian1 = [row for row in s if row[-1] == 1]
    X1 = [sublist[:-1] for sublist in qian1]
    # 使用列表推导式筛选出最后一列为0的子列表
    hou0 = [row for row in s if row[-1] == 0]
    X0 = [sublist[:-1] for sublist in hou0]
    X1 = np.array(X1)
    X0 = np.array(X0)
    # s = np.vstack((filtered_data, other_data))
    # print(s)
    # end = s.shape[1] - 1
    # X = s[:, :end]
    # y = s[:, -1]
    # X1=filtered_data[:, :-1]
    # X0=other_data[:, :-1]
    # y1=filtered_data[:, -1]
    # y0=other_data[:, -1]
    return X1,X0

#1.2.2
def Linear(X_train, X_test, y_train, y_test):
    clf = RidgeClassifier().fit(X_train, y_train)
    print( y_test)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    print(hun)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Linear***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn)+' '+str(sp)+'\n')
    f.close()
    y_score = clf.decision_function(X_test)
    print(type(y_score))
    print(type(y_test))
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.4.1
def SVC_T(X_train, X_test, y_train, y_test):
    from sklearn import svm, datasets
    random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    y_pred = svm.fit(X_train, y_train).predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***SVM***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    y_score = svm.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.5.1
def SGD(X_train, X_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***SGD***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.decision_function(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.6.2
def Neighbors(X_train, X_test, y_train, y_test):
    neigh = RadiusNeighborsClassifier(radius=1.0)
    neigh = neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***Neighbors***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = neigh.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.7.2
def GPC(X_train, X_test, y_train, y_test):
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train, y_train)
    y_pred = gpc.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0,0]/(hun[0,0]+hun[0,1])
    sp = hun[1,1]/(hun[1,1]+hun[1,0])
    print("***GPC***")
    print("sn = ",sn)
    print("sp = ",sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = gpc.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.9.1
def Gaussian_NB(X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Gaussian_NB***")
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    print("sn = ", sn)
    print("sp = ", sp)

    y_score = gnb.fit(X_train, y_train).predict_proba(X_test)
    y_score = y_score[:,1]# y_score是正类的概率估计值
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.9.4
def Bernoulli_NB(X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    bnb = BernoulliNB()
    y_pred = bnb.fit(X_train, y_train).predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Bernoulli_NB***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = bnb.fit(X_train, y_train).predict_proba(X_test)
    y_score = y_score[:, 1]  # y_score是正类的概率估计值
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.10.1
def DT(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #tree.plot_tree(clf)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***DT***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr,roc_auc

#1.11.1
def Bagging(X_train, X_test, y_train, y_test):
    clf = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***Bagging***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.2
def RandomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***RandomForest***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.3
def AdaBoost(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***AdaBoost***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.4
def GradientBoosting(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***GradientBoosting***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.5
def HistGradientBoosting(X_train, X_test, y_train, y_test):
    clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***HistGradientBoosting***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.17.2
def  MLPClassifier_1 (X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***MLPClassifier***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#定义class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2).double()
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(1, 100, num_layers=1, dropout=0.5,
                            bidirectional=True).double()
        self.liner1 = nn.Linear(200, 1).double()
        self.liner2 = nn.Linear(10, 2).double()

    def forward(self, x):
        output = self.conv1(x)
        output = self.max_pool1(output)
        w = output.shape[2]
        self.lstm = nn.LSTM(w, 100, num_layers=1, dropout=0.5,
                            bidirectional=True).double()
        hidden_cell = (torch.zeros([2, 10, 100], dtype=torch.double), torch.zeros([2, 10, 100], dtype=torch.double))
        # x.view(-1,40 * 14)
        lstm_out, (h_n, h_c) = self.lstm(output, hidden_cell)
        output = self.liner1(lstm_out)
        output = output.permute(0, 2, 1)
        output = self.liner2(output)
        output = output.squeeze(1)
        output = F.softmax(output)
        return output


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP, FP, TN, FN

# CNNBilstm
def CNNBilstm(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(Y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_test)), torch.from_numpy(np.array(Y_test)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
    test_loader = DataLoader(valid_data, shuffle=True, batch_size=8)

    # 建模三件套：loss，优化，epochs -
    model = Net()  # 模型
    loss_function = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 5
    # 开始训练
    model.train()
    for i in range(epochs):
        acc1 = []
        precision_scores = []
        f1_scores = []
        recall_scores = []
        sp1 = []
        MCC1 = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.unsqueeze(1).double())  # .squeeze()
            # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            acc1.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if ((TN + FP) != 0):
                Sp = TN / (TN + FP)
            else:
                Sp = 0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
        #print("Train Step:", i," acc:{:.6f}, pre:{:.4f},f1score:{:.4f},Sn:{:.4f},Sp:{:.4f},MCC:{:.4f} ".format(np.array(acc1).mean(),np.array(precision_scores).mean(),np.array(f1_scores).mean(),
        # np.array(recall_scores).mean(), np.array(sp1).mean(), np.array(MCC1).mean()))
    # 开始验证
    model.eval()
    acc2 = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    sp1 = []
    MCC1 = []
    p1 = []
    l1 = []

    for i in range(epochs):
        for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！
            y_pred = model(seq.unsqueeze(1).double())  # .squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            p = y_pred.detach().numpy().tolist()
            l = labels.numpy().tolist()
            for j in range(len(p)):

                p1.append(p[j])
                l1.append(int(l[j]))
            acc2.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
    TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    if ((TN + FP) != 0):
        Sp = TN / (TN + FP)
    else:
        Sp = 0
    sp1.append(Sp)
    MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    MCC1.append(MCC)
    print('***CNNBilstm***')
    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())

    fpr, tpr, threshold = roc_curve(l1, [y[1] for y in p1])  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

def CNNBilstm_Attention(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(Y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_test)), torch.from_numpy(np.array(Y_test)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
    test_loader = DataLoader(valid_data, shuffle=True, batch_size=8)

    # 建模三件套：loss，优化，epochs -
    model = Net()  # 模型
    loss_function = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 5
    # 开始训练
    model.train()
    for i in range(epochs):
        acc1=[]
        precision_scores = []
        f1_scores=[]
        recall_scores=[]
        sp1=[]
        MCC1=[]
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()
            # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            acc1.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if((TN+FP)!=0):
              Sp = TN / (TN + FP)
            else:
              Sp=0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
    # 开始验证
    model.eval()
    acc2 = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    sp1 = []
    MCC1 = []
    p1 = []
    l1 = []
    y_score =[]
    for i in range(epochs):
        for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            p = torch.max(y_pred, dim=1)[1].numpy().tolist()
            l = labels.numpy().tolist()
            for j in range(len(p)):
                p1.append(p[j])
                l1.append(int(l[j]))
            acc2.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
    TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    if ((TN + FP) != 0):
        Sp = TN / (TN + FP)
    else:
        Sp = 0
    sp1.append(Sp)
    MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    MCC1.append(MCC)
    print('***CNNBilstm_Attention***')
    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())

    fpr, tpr, threshold = roc_curve(l1,p1)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#TextCNN
def TextCNN(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(Y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_test)), torch.from_numpy(np.array(Y_test)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
    test_loader = DataLoader(valid_data, shuffle=True, batch_size=8)
    # 建模三件套：loss，优化，epochs -
    model = Net()  # 模型
    loss_function = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 5
    # 开始训练
    model.train()
    for i in range(epochs):
        acc1=[]
        precision_scores = []
        f1_scores=[]
        recall_scores=[]
        sp1=[]
        MCC1=[]
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()
            # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            acc1.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if((TN+FP)!=0):
              Sp = TN / (TN + FP)
            else:
              Sp=0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
    # 开始验证
    model.eval()
    acc2 = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    sp1 = []
    MCC1 = []
    p1 = []
    l1 = []
    for i in range(epochs):
        for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            p = torch.max(y_pred, dim=1)[1].numpy().tolist()
            l = labels.numpy().tolist()
            for j in range(len(p)):
                p1.append(p[j])
                l1.append(int(l[j]))
            acc2.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
    TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    if ((TN + FP) != 0):
        Sp = TN / (TN + FP)
    else:
        Sp = 0
    sp1.append(Sp)
    MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    MCC1.append(MCC)
    print('***TextCNN***')

    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())

    fpr, tpr, threshold = roc_curve(l1, p1)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#Graph_Code
class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, classes)

    def forward(self, data):
        x, edge_index = data['x'],data['edge_index']
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GraphSAGE_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(features, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data['x'],data['edge_index']
        x = self.sage1(x, edge_index).float()
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index).float()
        return F.log_softmax(x, dim=1)
class GCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(features, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data['x'],data['edge_index']
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class Graph_Class(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.line1 =nn.Linear(64, output_size)
  def forward(self, data):
    # 1. Obtain node embeddings
    x = self.conv1(data['x'], data['edge_index'])
    x = self.norm2(x)
    x = torch.relu(x)
    x = self.gat2(x, data['edge_index'])
    x = self.norm1(x)
    x = torch.relu(x)
    x = F.dropout(x, p=0.6, training=self.training)
    x = torch.sigmoid(self.line1(x))
    return x

def Graph_Code(fname):
    warnings.filterwarnings('ignore')

    idx_features_labels = np.genfromtxt(fname,
                                        dtype=np.dtype(str))
    data = {}
    data['x'] = Parameter(torch.from_numpy(np.float32(idx_features_labels[:, 0:-1])))
    data['label'] = torch.tensor(torch.from_numpy(np.float32(idx_features_labels[:, -1])))
    w = idx_features_labels.shape
    num_node_features = w[1] - 1    #这个参数是维度也就是列数
    num_classes = 2
    x1 = torch.arange(w[0], dtype=torch.int64)      #这边输入的是行数
    x2 = torch.arange(w[0], dtype=torch.int64)
    data['edge_index'] = torch.stack([x1, x2], 0)
    # print(data['edge_index'])
    # print(data['edge_index'].dtype)
    train_mask = range(0, w[0])             #这是训练集的范围就是 0行到多少行
    val_mask = range(w[1] - 1, w[0])
    test_mask = range(w[1] - 1, w[0])       #测试集就是  w[1] - 1 行 到 w[0]行


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Graph_Class(num_node_features, 16, num_classes).float()#.to(device)/
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # main loop
    dur = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    sp1=[]
    p2=[]
    l2=[]
    MCC1 = []
    # batch=torch.tensor(1)
    for epoch in range(100):
        if epoch >= 3:
            t0 = time.time()
        best_val_acc = 0
        logits = model(data)
        logp = F.log_softmax(logits, 1)
        pred = F.log_softmax(logits, 1).argmax(1)
        loss = F.nll_loss(logp[train_mask], data['label'][train_mask].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask].long())
        train_acc = (pred[train_mask] == data['label'][train_mask]).float().mean()
        val_acc = (pred[val_mask] == data['label'][test_mask]).float().mean()
        test_acc = (pred[test_mask] == data['label'][test_mask]).float().mean()
        MCC =0
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        logit = logits.detach().numpy().tolist()
        p = pred[test_mask].numpy().tolist()
        l = data['label'][test_mask].numpy().tolist()
        vali_f1 = f1_score(l, p, average="micro")
        recall=recall_score(l, p, labels=None, pos_label=1, average='binary', sample_weight = None)
        p1=logp[test_mask].detach().numpy().tolist()
        for i in range(len(p1)):
            p2.append(p1[i])
            l2.append(int(l[i]))
        if epoch >= 3:
            dur.append(time.time() - t0)
        TP, FP, TN, FN = perf_measure(l, p)
        # Sp = TN/(TN+FP)
        if ((TN + FP) != 0):
            Sp = TN / (TN + FP)
        else:
            Sp = 0
        sp1.append(Sp)
        MCC = matthews_corrcoef(l, p)
        accuracy_scores.append(accuracy_score(l, p))
        precision_scores.append(precision_score(l, p))
        recall_scores.append(recall_score(l, p))
        f1_scores.append(f1_score(l, p))
        #print("Epoch {:05d} | Time(s) {:.4f} | train acc: {:.6f}| test acc: {:.6f}| f1_score: {:.5f}| Sn: {:.5f}| MCC: {:.5f}".format(
            #epoch+1, np.mean(dur),train_acc,test_acc,vali_f1,recall,MCC))

    # TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    # if ((TN + FP) != 0):
    #     Sp = TN / (TN + FP)
    # else:
    #     Sp = 0
    # sp1.append(Sp)
    # MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    # MCC1.append(MCC)
    print('***GraphCode***')

    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())
    if np.array(recall_scores).mean()+np.array(sp1).mean()!=0:
        fpr, tpr, threshold = roc_curve(l2, [y[1] for y in p2])  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        return fpr, tpr, roc_auc

def XGboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    print(hun)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***XGboosting***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn)+' '+str(sp)+'\n')
    f.close()
    y_pred_prob = model.predict_proba(X_test)[:, 1]#这边用了y_pred_prob代替了y_score因为不是sklearn库。
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    return fpr, tpr, roc_auc


def lightgbm(X_train, X_test, y_train, y_test):
    ## 定义 LightGBM 模型
    clf = LGBMClassifier()
    # 在训练集上训练LightGBM模型
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Lightgbm***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()

    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#BLS-CroSS-validation
def show_accuracy(predictLabel, Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count / len(Label), 5))

class node_generator(object):
    def __init__(self, isenhance=False):
        self.Wlist = []
        self.blist = []
        self.function_num = 0
        self.isenhance = isenhance

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def linear(self, x):
        return x

    def orth(self, W):
        """
        orth是正交基的意思，求正交基可能是为了使增强节点彼此无关
        目前看来，这个函数应该配合下一个generator函数是生成权重的
        此函数传入的weights与传出的weights的shape是一样的。
        """
        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T
                w_sum += (w.T.dot(wj))[0, 0] * wj
            w -= w_sum
            w = w / np.sqrt(w.T.dot(w))
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        for i in range(times):
            W = 2 * np.random.random(size=shape) - 1
            if self.isenhance == True:
                W = self.orth(W)  # 只在增强层使用
            b = 2 * np.random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, function_num):
        # 按照bls的理论，mapping layer是输入乘以不同的权重加上不同的偏差之后得到的
        # 若干组，所以，权重是一个列表，每一个元素可作为权重与输入相乘
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.function_num = {'linear': self.linear,
                             'sigmoid': self.sigmoid,
                             'tanh': self.tanh,
                             'relu': self.relu}[function_num]  # 激活函数供不同的层选择
        # 下面就是先得到一组mapping nodes，再不断叠加，得到len(Wlist)组mapping nodes
        nodes = self.function_num(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.function_num(data.dot(self.Wlist[i]) + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.function_num(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.function_num(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / (self._std + 0.001)

    def transform(self, testdata):
        return (testdata - self._mean) / (self._std + 0.001)

class broadNet(object):
    def __init__(self, map_num=10, enhance_num=10, map_function='linear', enhance_function='linear', batchsize='auto'):
        self.map_num = map_num
        self.enhance_num = enhance_num
        self.batchsize = batchsize
        self.map_function = map_function
        self.enhance_function = enhance_function

        self.W = 0
        self.pseudoinverse = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhance_generator = node_generator(isenhance=True)

    def fit(self, data, label):
        if self.batchsize == 'auto':
            self.batchsize = data.shape[1]

        data = self.normalscaler.fit_transform(data)
        label = self.onehotencoder.fit_transform(np.mat(label).T)

        mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize, self.map_function)
        enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num, self.batchsize,
                                                             self.enhance_function)
        #
        # print('number of mapping nodes {0}, number of enhence nodes {1}'.format(mappingdata.shape[1],
        #                                                                         enhancedata.shape[1]))
        # print('mapping nodes maxvalue {0} minvalue {1} '.format(round(np.max(mappingdata), 5),
        #                                                         round(np.min(mappingdata), 5)))
        # print('enhence nodes maxvalue {0} minvalue {1} '.format(round(np.max(enhancedata), 5),
        #                                                         round(np.min(enhancedata), 5)))

        inputdata = np.column_stack((mappingdata, enhancedata))
        # print('input shape ', inputdata.shape)
        pseudoinverse = np.linalg.pinv(inputdata)
        # 新的输入到输出的权重
        # print('pseudoinverse shape:', pseudoinverse.shape)
        self.W = pseudoinverse.dot(label)

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def decode1(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis[1])
        return np.array(Y)

    def accuracy(self, predictlabel, label):
        label = np.ravel(label).tolist()
        predictlabel = predictlabel.tolist()
        count = 0
        for i in range(len(label)):
            if label[i] == predictlabel[i]:
                count += 1
        return (round(count / len(label), 5))

    def predict(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhancedata = self.enhance_generator.transform(test_mappingdata)

        test_inputdata = np.column_stack((test_mappingdata, test_enhancedata))
        return self.decode(test_inputdata.dot(self.W))
    def predict_score(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhancedata = self.enhance_generator.transform(test_mappingdata)

        test_inputdata = np.column_stack((test_mappingdata, test_enhancedata))
        return self.decode1(test_inputdata.dot(self.W))

def BLS_Cross(traindata, testdata, trainlabel, testlabel):

    label = trainlabel
    print(label)
    data = traindata
    print(data)


    bls = broadNet(map_num=10,
                   enhance_num=10,
                   map_function='relu',
                   enhance_function='relu',
                   batchsize=10)

    starttime = datetime.datetime.now()
    bls.fit(traindata, trainlabel)
    endtime = datetime.datetime.now()
    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))
    predictlabel = bls.predict(testdata)
    # print(show_accuracy(predictlabel, testlabel))
    Training_Division_prec = precision_score(testlabel, predictlabel, pos_label=1)
    Training_Division_recall = recall_score(testlabel, predictlabel, pos_label=1)
    Training_Division_f1 = f1_score(testlabel, predictlabel, pos_label=1)
    # test_auc = roc_auc_score(text_label, lr_score[:, 1])
    con_matrix = confusion_matrix(testlabel, predictlabel)
    print('--con_matrix',con_matrix)
    print('testlabel',testlabel)#17个数
    print('predictlabel',predictlabel)
    Training_Division_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    Training_Division_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (
            ((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (
                        con_matrix[0][0] + con_matrix[0][1]) * (con_matrix[0][0] + con_matrix[1][0])) ** 0.5)
    print("Training Set Division  :acc: ", show_accuracy(predictlabel, testlabel), " ; prec: ",  Training_Division_spec, " ; recall: ", Training_Division_recall ,
          " ; f1: ", Training_Division_f1, "  ; spec:", Training_Division_spec, " ; mcc: ", Training_Division_mcc)
    #十折交叉验证
    KF = KFold(n_splits=10, shuffle=True, random_state=100)
    Pre=[]
    Acc=[]
    Sp=[]
    Sn=[]
    F1=[]
    p=[]
    l=[]
    MCC=[]
    for train_index, test_index in KF.split(data):
        bls.fit(data[train_index], label[train_index])
        predictlabel = bls.predict(data[test_index])
        predictscore = bls.predict_score(data[test_index])
        k= predictscore.tolist()
        m=label[test_index].tolist()
        for i in range(len(m)):
            p.append(k[i])
            l.append(m[i])
        Acc.append(show_accuracy(predictlabel, label[test_index]))
        Training_Division_prec = precision_score(label[test_index], predictlabel, pos_label=1)
        Training_Division_recall = recall_score(label[test_index], predictlabel, pos_label=1)
        Training_Division_f1 = f1_score(label[test_index], predictlabel, pos_label=1)
        print('++label[test_index]',label[test_index])
        print('predictlabel', predictlabel)
        con_matrix = confusion_matrix(label[test_index], predictlabel)
        print('con_matrix',con_matrix)
        print('type',type(con_matrix))
        if len(con_matrix)==1:
            new_array = np.zeros((2, 2))
            new_array[0, 0] = con_matrix[0, 0]
            Training_Division_spec = new_array[0][0] / (new_array[0][0] + new_array[0][1])
            Training_Division_mcc = (new_array[0][0] * new_array[1][1] - new_array[0][1] * new_array[1][0]) / (
                    ((new_array[1][1] + new_array[0][1]) * (new_array[1][1] + new_array[1][0]) * (
                            new_array[0][0] + new_array[0][1]) * (new_array[0][0] + new_array[1][0])) ** 0.5)
            Sp.append(Training_Division_spec)
            Pre.append(Training_Division_prec)
            Sn.append(Training_Division_recall)
            F1.append(Training_Division_f1)
            MCC.append(Training_Division_mcc)
        else:
            Training_Division_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
            Training_Division_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (
                    ((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (
                            con_matrix[0][0] + con_matrix[0][1]) * (con_matrix[0][0] + con_matrix[1][0])) ** 0.5)
            Sp.append(Training_Division_spec)
            Pre.append(Training_Division_prec)
            Sn.append(Training_Division_recall)
            F1.append(Training_Division_f1)
            MCC.append(Training_Division_mcc)
    print("Training Set Cross-validation  :acc: ", np.mean(Acc), " ; prec: ", np.mean(Pre)," ; recall: ", np.mean(Sn), " ; f1: ",   np.mean(F1), "  ; spec:", np.mean(Sp), " ; mcc: ", np.mean(MCC))
    f = open(fnameresult, 'a')
    f.write(str(np.mean(Sn)) + ' ' + str(np.mean(Sp)) + '\n')
    f.close()
    fpr, tpr, threshold = roc_curve(l, p)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

def Cascade_Forest1(traindata, testdata, trainlabel, testlabel):
    model = CascadeForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    Linear_hun = metrics.confusion_matrix(y_test, y_pred)
    sn = Linear_hun[0, 0] / (Linear_hun[0, 0] + Linear_hun[0, 1])
    sp = Linear_hun[1, 1] / (Linear_hun[1, 1] + Linear_hun[1, 0])
    print("***Cascade Forest***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = model.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc


def Roc(name,fpr, tpr,roc_auc):
    lw = 2
    #宽度
    plt.plot(fpr, tpr,
         lw=lw, label=name+' ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return plt.figure




if __name__ == "__main__":

    feature = ['hongse','huise','lanse','lvse','huangse','allcolor']#
    for feature in feature:

        fname = 'D:/a/data/Kouqiang_0607_KG_SHAP/' + feature + '_shap.txt'


        X1,X0 = dataread(fname) #交叉验证导入
        y1=np.ones(len(X1))
        y0=np.zeros(len(X0))

        kf = KFold(n_splits=10, shuffle=False, random_state=None)


        AA=1
        print(type(X1))
        # # 划分数据集
        for (train_index_0, test_index_0), (train_index_1, test_index_1) in zip(kf.split(X0), kf.split(X1)):

            X0_train, X0_test = X0[train_index_0], X0[test_index_0]
            y0_train, y0_test = y0[train_index_0], y0[test_index_0]

            X1_train, X1_test = X1[train_index_1], X1[test_index_1]
            y1_train, y1_test = y1[train_index_1], y1[test_index_1]

            X_train = np.vstack((X0_train, X1_train))
            X_test = np.vstack((X0_test, X1_test))
            y_train = np.hstack((y0_train, y1_train))
            y_test = np.hstack((y0_test, y1_test))


            fnameresult = 'D:/a/result/test/txt/Kouqiang_0607_KG_SHAP/' + feature +str(AA)+ '_shap_results.txt'#
            fnamefig = 'D:/a/result/test/fig/Kouqiang_0607_KG_SHAP/' + feature+str(AA) +'_shap.png'#+ str(AA)
            print(u'start 当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))


            Linear_fpr, Linear_tpr, Linear_roc_auc = Linear(X_train, X_test, y_train, y_test)
            #   SVC_fpr, SVC_tpr, SVC_roc_auc = SVC_T(X_train, X_test, y_train, y_test)
            CF_fpr, CF_tpr, CF_roc_auc = Cascade_Forest1(X_train, X_test, y_train, y_test)
            print(u'svc 当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            SGD_fpr, SGD_tpr, SGD_roc_auc = SGD(X_train, X_test, y_train, y_test)
            # Neighbors_fpr, Neighbors_tpr, Neighbors_roc_auc = Neighbors(X_train, X_test, y_train, y_test)
            GPC_fpr, GPC_tpr, GPC_roc_auc = GPC(X_train, X_test, y_train, y_test)
            GaussianNB_fpr, GaussianNB_tpr, GaussianNB_roc_auc = Gaussian_NB(X_train, X_test, y_train, y_test)
            Bernoulli_fpr, Bernoulli_tpr, Bernoulli_roc_auc = Bernoulli_NB(X_train, X_test, y_train, y_test)
            Tree_fpr, Tree_tpr, Tree_roc_auc = DT(X_train, X_test, y_train, y_test)
            Bagging_fpr, Bagging_tpr, Bagging_roc_auc = Bagging(X_train, X_test, y_train, y_test)
            RandomForest_fpr, RandomForest_tpr, RandomForest_roc_auc = RandomForest(X_train, X_test, y_train, y_test)
            AdaBoost_fpr, AdaBoost_tpr, AdaBoost_roc_auc = AdaBoost(X_train, X_test, y_train, y_test)
            GradientBoosting_fpr, GradientBoosting_tpr, GradientBoosting_roc_auc = GradientBoosting(X_train, X_test, y_train,                                                                                                y_test)
            HGB_fpr, HGB_tpr, HGB_roc_auc = HistGradientBoosting(X_train, X_test, y_train, y_test)
            # CNNB_fpr, CNNB_tpr, CNNB_roc_auc = CNNBilstm(X_train, X_test, y_train, y_test)
            # CNNB_A_fpr, CNNB_A_tpr, CNNB_A_roc_auc = CNNBilstm_Attention(X_train, X_test, y_train, y_test)
            # TextCNN_fpr, TextCNN_tpr, TextCNN_roc_auc = TextCNN(X_train, X_test, y_train, y_test)
            # Graph_Code_fpr, Graph_Code_tpr, Graph_Code_roc_auc = Graph_Code(fname)
            MLPC_fpr, MLPC_tpr, MLPC_roc_auc = MLPClassifier_1(X_train, X_test, y_train, y_test)
            XG_fpr, XG_tpr, XG_roc_auc = XGboost(X_train, X_test, y_train, y_test)
            GBM_fpr, GBM_tpr, GBM_roc_auc = lightgbm(X_train, X_test, y_train, y_test)
            BLS_Cross_fpr, BLS_Cross_tpr, BLS_Cross_roc_auc  =BLS_Cross(X_train, X_test, y_train, y_test)

            # 绘图
            plt.figure(figsize=(10, 10))
            Roc("Linear Discriminant", Linear_fpr, Linear_tpr, Linear_roc_auc) #！
            #  Roc("Support Vector Machine", SVC_fpr, SVC_tpr, SVC_roc_auc)
            Roc("Cascade Forest", CF_fpr, CF_tpr, CF_roc_auc) #！
            Roc("Stochastic Gradient Descent", SGD_fpr, SGD_tpr, SGD_roc_auc)#！
            # Roc("K Nearest Neighbors",Neighbors_fpr, Neighbors_tpr, Neighbors_roc_auc)
            Roc("Gaussian Processes", GPC_fpr, GPC_tpr, GPC_roc_auc)#！
            Roc("Gaussian Naive Bayes", GaussianNB_fpr, GaussianNB_tpr, GaussianNB_roc_auc)#！
            Roc("Bernoulli Naive Bayes", Bernoulli_fpr, Bernoulli_tpr, Bernoulli_roc_auc)#！
            Roc("Decision Tree", Tree_fpr, Tree_tpr, Tree_roc_auc)#!
            Roc("Bagging", Bagging_fpr, Bagging_tpr, Bagging_roc_auc)#!
            Roc("Random Forest", RandomForest_fpr, RandomForest_tpr, RandomForest_roc_auc)#!
            Roc("AdaBoost", AdaBoost_fpr, AdaBoost_tpr, AdaBoost_roc_auc)#!
            Roc("Gradient Boosting", GradientBoosting_fpr, GradientBoosting_tpr, GradientBoosting_roc_auc)#!
            Roc("Hist Gradient Boosting", HGB_fpr, HGB_tpr, HGB_roc_auc)#!
            # Roc("CNNBilstm", CNNB_fpr, CNNB_tpr, CNNB_roc_auc)
            # Roc("CNNBilstm_Attention", CNNB_A_fpr, CNNB_A_tpr, CNNB_A_roc_auc)
            # Roc("TextCNN", TextCNN_fpr, TextCNN_tpr, TextCNN_roc_auc)
            # Roc("Graph_Code", Graph_Code_fpr, Graph_Code_tpr, Graph_Code_roc_auc)
            Roc("MLPClassifier_1", MLPC_fpr, MLPC_tpr, MLPC_roc_auc)#!
            Roc("XGboosting", XG_fpr, XG_tpr, XG_roc_auc)#!
            Roc("LightGBM", GBM_fpr, GBM_tpr, GBM_roc_auc)#!
            Roc("BLS", BLS_Cross_fpr, BLS_Cross_tpr, BLS_Cross_roc_auc)#!

            # Roc("svm", fpr, tpr, roc_auc)
            plt.savefig(fnamefig, dpi=1000, bbox_inches='tight')
            print(u'end 当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            AA = AA + 1




        #plt.show()