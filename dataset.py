from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt


# 生成逻辑回归数据集
def generate_logistic_regression_data(samples=100, features=2, classes=2):
    """
    生成一个用于逻辑回归的二分类问题数据集。

    参数:
    - samples: 要生成的样本数量
    - features: 每个样本的特征数量
    - classes: 类别数（二分类）

    返回:
    - X: 生成的样本特征矩阵
    - y: 每个样本对应的类别标签
    """

    X, y = make_classification(n_samples=samples,
                               n_features=features,
                               n_classes=classes,
                               n_clusters_per_class=1,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               random_state=42)

    return X, y

if __name__ == '__main__':
    # 生成数据
    X, y = generate_logistic_regression_data(samples=200, features=2, classes=2)
    print(X.shape,y.shape)
    #绘制数据点
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    plt.title('Generated Logistic Regression Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()