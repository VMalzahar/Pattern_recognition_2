import numpy as np
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from LDA import lda
from KNN import KNN


# 读取数据集
data = io.loadmat('Yale_64x64.mat')
ins_perclass, class_number, train_test_split = 11, 15, 9
input_dim = data['fea'].shape[1]

# 数据重塑
feat = data['fea'].reshape(-1, ins_perclass, input_dim)
label = data['gnd'].reshape(-1, ins_perclass)

# 划分训练集和测试集
train_data = feat[:, :train_test_split, :].reshape(-1, input_dim)
test_data = feat[:, train_test_split:, :].reshape(-1, input_dim)
train_label = label[:, :train_test_split].reshape(-1)
test_label = label[:, train_test_split:].reshape(-1)

# 降维
lda_2d = lda(train_data, train_label , 2)
train_data_lda_2d = train_data @ lda_2d
test_data_lda_2d = test_data @ lda_2d

# 训练KNN分类器
knn = KNN(3)  # 选择k值为3
knn.fit(train_data_lda_2d, train_label)

# 对测试数据进行预测
predictions = knn.predict(test_data_lda_2d)

print(classification_report(test_label, predictions))
