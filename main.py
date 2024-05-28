import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

# Apply PCA
pca = PCA(n_components=8)
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)

# Apply LDA
lda = LDA(n_components=8)
lda.fit(train_data, train_label)
train_data_lda = lda.transform(train_data)
test_data_lda = lda.transform(test_data)

# # Reshape and display the first 8 principal components as images
# def plot_components(components, title):
#     fig, axes = plt.subplots(1, 8, figsize=(20, 4))
#     for i, ax in enumerate(axes):
#         ax.imshow(components[i].reshape(64, 64), cmap='gray')
#         ax.axis('off')
#     plt.suptitle(title)
#     plt.show()


def plot_components(components, title):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(components[i].reshape(64, 64), cmap='gray')
        ax.set_title(f'Component {i+1}')
        ax.axis('off')
    fig.suptitle(title)
    plt.show()

plot_components(pca.components_, "First 8 PCA Components")
plot_components(lda.scalings_.T, "First 8 LDA Components")


# 可视化降维到2维的数据
def plot_2d(data, labels, title):
    plt.figure(figsize=(10, 8))
    for i in np.unique(labels):
        subset = data[labels == i]
        plt.scatter(subset[:, 0], subset[:, 1], label=f'Class {i}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.legend()
    plt.show()

# 仅展示三类数据的可视化
# plot_classes = [1, 2, 3]
reduced_dim = 2

pca_2d = PCA(n_components=reduced_dim)
lda_2d = LDA(n_components=reduced_dim)

train_data_pca_2d = pca_2d.fit_transform(train_data)
test_data_pca_2d = pca_2d.transform(test_data)

train_data_lda_2d = lda_2d.fit_transform(train_data, train_label)
test_data_lda_2d = lda_2d.transform(test_data)


train_indices = train_label
test_indices = test_label

plot_2d(train_data_pca_2d, train_label, 'PCA - Training Data')
plot_2d(test_data_pca_2d, test_label, 'PCA - Testing Data')

plot_2d(train_data_lda_2d, train_label, 'LDA - Training Data')
plot_2d(test_data_lda_2d, test_label, 'LDA - Testing Data')
# Dimensionality reduction to 2D for visualization

# # Function to visualize the data
# def plot_2d_data(data, labels, title):
#     plt.figure(figsize=(10, 6))
#     for i in np.unique(labels):
#         plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Class {i}')
#     plt.title(title)
#     plt.xlabel('Component 1')
#     plt.ylabel('Component 2')
#     plt.legend()
#     plt.show()

# # Visualize the PCA and LDA reduced data
# plot_2d_data(train_data_pca_2d, train_label, "PCA - Training Data")
# plot_2d_data(test_data_pca_2d, test_label, "PCA - Testing Data")
# plot_2d_data(train_data_lda_2d, train_label, "LDA - Training Data")
# plot_2d_data(test_data_lda_2d, test_label, "LDA - Testing Data")
