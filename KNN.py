import numpy as np

class KNN:
    def __init__(self, k=3):
        # 初始化KNN分类器
        self.k = k

    def fit(self, X, y):
        # 训练模型
        self.date = X
        self.label = y

    def predict(self, X):
        # 预测新样本的类别
        y_pred = [self.single_predict(x) for x in X]
        return np.array(y_pred)

    def single_predict(self, x):
        # 预测单个样本的类别
        distances = [np.linalg.norm(x - x_) for x_ in self.date]
        ids = np.argsort(distances)[:self.k]
        labels = [self.label[i] for i in ids]
        return np.bincount(labels).argmax()
         
