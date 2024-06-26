import copy
import random

import matplotlib.pyplot as plt

import kernel_func
import numpy as np

class SVM_Classifier:
    def __init__(self, C=1.0, gamma=1.0, degree=3, coef0=1.0, kernel=None, kkt_tol=1e-3, alpha_tol=1e-3,
                 max_epochs=100):
        self.C = C
        self.kernel = kernel  # 选择的核函数
        if self.kernel is None or self.kernel.lower() == "linear":
            self.kernel_func = kernel_func.linear()  # self.kernel_func(x_i, x_j)
        elif self.kernel.lower() == "poly":
            self.kernel_func = kernel_func.poly(degree, coef0)  # self.kernel_func(x_i, x_j)
        elif self.kernel.lower() == "rbf":
            self.kernel_func = kernel_func.rbf(gamma)
        else:
            print("仅限linear、poly或rbf，值为None则默认为Linear线性核函数...")
            self.kernel_func = kernel_func.linear()

        self.alpha_tol = alpha_tol  # 支持向量容忍度

        self.kkt_tol = kkt_tol  # 在精度内检查
        self.max_epochs = max_epochs
        self.alpha = None  # 松弛变量
        self.E = None  # 误差
        self.w, self.b = None, None  # SVM的模型系数

        self.support_vectors = []  # 记录支持向量的索引
        self.support_vectors_alpha = []  # 支持向量所对应的松弛变量
        self.support_vectors_x, self.support_vectors_y = [], []  # 支持向量所对应的样本和目标
        self.cross_entropy_loss = []  # 优化过程中的交叉熵损失

    def init_params(self, x_train, y_train):
        """
        初始化必要参数
        :param x_train: 训练集
        :param y_train: 编码后的目标集
        :return:
        """
        n_samples, n_features = x_train.shape
        self.w, self.b = np.zeros(n_features), 0.0  # 模型系数初始化为0值
        self.alpha = np.zeros(n_samples)  # 松弛变量
        self.E = self.decision_func(x_train) - y_train  # 初始化误差，所有样本类别取反

    def decision_func(self, x):
        """
        SVM模型的预测计算，
        :param x: 可以是样本集，也可以是单个样本
        :return:
        """
        if len(self.support_vectors) == 0:  # 当前没有支持向量
            if x.ndim <= 1:  # 标量或单个样本
                return 0
            else:
                return np.zeros(x.shape[0])  # np.zeros(x.shape[:-1])
        else:
            if x.ndim <= 1:
                wt_x = 0.0  # 模型w^T * x + b的第一项求和
            else:
                wt_x = np.zeros(x.shape[0])
            for i in range(len(self.support_vectors)):
                # 公式：w^T * x = sum(alpha_i * y_i * k(xi, x))
                wt_x += self.support_vectors_alpha[i] * self.support_vectors_y[i] * \
                        self.kernel_func(x, self.support_vectors_x[i])
        return wt_x + self.b
    