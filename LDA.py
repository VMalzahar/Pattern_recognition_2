import numpy as np

def lda(X, y):
    """
    执行线性判别分析（LDA）
    
    参数:
    X -- 输入数据矩阵，每行是一个样本
    y -- 样本的类别标签
    
    返回:
    W -- 投影矩阵
    """
    # 获取类别
    classes = np.unique(y)
    
    # 计算每个类别的均值向量
    mean_vectors = []
    for cls in classes:
        mean_vectors.append(np.mean(X[y == cls], axis=0))
    mean_vectors = np.array(mean_vectors)
    
    # 计算类内散度矩阵 Sw
    n_features = X.shape[1]
    Sw = np.zeros((n_features, n_features))
    for cls, mean_vec in zip(classes, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))
        for row in X[y == cls]:
            row, mean_vec = row.reshape(n_features, 1), mean_vec.reshape(n_features, 1)
            class_sc_mat += (row - mean_vec).dot((row - mean_vec).T)
        Sw += class_sc_mat
    
    # 计算总均值向量
    overall_mean = np.mean(X, axis=0)
    
    # 计算类间散度矩阵 Sb
    Sb = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == classes[i]].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    
    # 计算 Sw^-1 * Sb
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    
    # 选择前 k 个特征向量（对应于最大的 k 个特征值）
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    W = np.hstack([eig_pairs[i][1].reshape(n_features, 1) for i in range(len(classes) - 1)])
    
    return W

# 测试 LDA 函数
if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    W = lda(X, y)
    X_lda = X.dot(W)
    
    print("投影矩阵 W:\n", W)
    print("降维后的数据:\n", X_lda)
