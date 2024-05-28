import numpy as np

def pca(X, n_components):
    # 标准化数据
    X_mean = np.mean(X, axis=0)
    X_std = (X - X_mean) / np.std(X, axis=0)
    
    # 计算协方差矩阵
    cov_matrix = np.cov(X_std, rowvar=False)
    
    # 计算特征值和特征向量
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
    
    # 按特征值从大到小排序
    sorted_index = np.argsort(eig_values)[::-1]
    sorted_eigenvectors = eig_vectors[:, sorted_index]
    
    # 选择前n_components个特征向量
    eig_vectors_subset = sorted_eigenvectors[:, :n_components]
    
    # 投影数据
    X_reduced = np.dot(eig_vectors_subset.T, X_std.T).T
    
    return X_reduced

if __name__ == '__main__':
# 示例数据
    X = np.array([[2.5, 2.4],
                [0.5, 0.7],
                [2.2, 2.9],
                [1.9, 2.2],
                [3.1, 3.0],
                [2.3, 2.7],
                [2.0, 1.6],
                [1.0, 1.1],
                [1.5, 1.6],
                [1.1, 0.9]])

    # 进行PCA，降维到1维
    X_pca = pca(X, n_components=1)
    print(X_pca)
