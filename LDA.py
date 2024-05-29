import numpy as np

def lda(data, target, n_dim=None):
    """
    线性判别分析(LDA)

    参数:
    data -- 输入数据，每行表示一个样本，每列表示一个特征
    target -- 样本标签，用于区分不同类别
    n_dim -- 输出数据的维度，默认为类别数，不能大于类别数

    返回:
    w -- 投影矩阵，可以用于对数据降维。
    """
    
    # 获取类别数
    clusters = np.unique(target)

    # 如果未指定输出维度，则设为类别数
    if n_dim is None:
        n_dim = len(clusters)

    # 检查输出维度是否合法
    if n_dim > len(clusters):
        raise ValueError("n_dim too large")

    # 计算类内散度矩阵Sw
    Sw = np.zeros((data.shape[1], data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai - datai.mean(axis=0)
        Sw += datai.T @ datai

    # 计算类间散度矩阵SB
    Sb = np.zeros((data.shape[1], data.shape[1]))
    u = data.mean(axis=0)
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(axis=0)
        Sb += Ni * (ui - u).reshape(-1, 1) @ (ui - u).reshape(1, -1)

    # 计算投影矩阵w
    S = np.linalg.inv(Sw) @ Sb
    eig_values, eig_vectors = np.linalg.eigh(S)
    
    sorted_indices = np.argsort(-eig_values)[:n_dim]
    w = eig_vectors[:, sorted_indices]
    return w
