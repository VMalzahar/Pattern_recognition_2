import numpy as np

def pca(data, n_dim):
    """
    主成分分析(PCA)

    参数:
    data -- 输入数据，每行表示一个样本，每列表示一个特征
    n_dim -- 输出数据的维度，即要保留的主成分个数

    返回:
    w -- 投影矩阵，可以用于数据降维
    """

    # 中心化处理
    data = data - np.mean(data, axis=0)
    
    # 计算协方差矩阵
    cov = np.cov(data, rowvar=False)
    
    # 计算协方差矩阵的特征值和特征向量
    eig_values, eig_vectors = np.linalg.eigh(cov)
    
    # 对特征值进行排序，并选取前n_dim个特征向量
    sorted_indices = np.argsort(-eig_values)[:n_dim]
    w = eig_vectors[:, sorted_indices]
    
    return w
