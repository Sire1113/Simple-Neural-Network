from dataset import generate_logistic_regression_data
import numpy as np
import math
import matplotlib.pyplot as plt


class Dense:
    def __init__(self, num_units, activation_func):
        self.num_units = num_units
        self.activation_func = activation_func
        self.Wn = None
        self.bn = None

    def forward(self, _input):
        self.A_in = _input  # A_in是输出本层的A,A是进入下一层的A
        if self.Wn is None or self.bn is None:
            # input每一列 为1组数据   每一行 为同一特征
            # input.shape[0]获取输入的特征数量
            features = _input.shape[0]
            # 根据单元数和输入特征数量初始化权重
            self.Wn = np.random.randn(self.num_units, features)
            # 根据单元数初始化偏差
            self.bn = np.zeros((self.num_units, 1))
        self.Z = self.compute_Z(self.Wn, _input, self.bn)
        self.A = self.activation(self.Z, self.activation_func)
        return self.A

    def backward(self, dZ_input, dW_input):
        m = dZ_input.shape[1]  # 输出的dZ是这一层的,input的是后一层的dZ
        self.dZn = np.dot(dW_input.T, dZ_input) * Dense.relu_derivative(self.Z)  # 默认使用ReLU
        self.dWn = np.dot(self.dZn, self.A_in.T) / m  # 这里也有问题,应该是后一层的dw与dz相乘
        self.dbn = np.sum(dZ_input, axis=1, keepdims=True) / m
        return self.dZn, self.dWn

    def activation(self, Z, activation_func):
        output = activation_func(Z)
        return output

    # ----------------------激活函数---------------------------
    @classmethod
    def relu(cls, Z):
        np.random.randn(1, 1)
        return np.maximum(Z, 0)

    @classmethod
    def sigmoid(cls, Z):
        return 1 / (1 + np.exp(-Z))

    @classmethod
    def linear(cls, Z):
        return Z

    @classmethod
    def tanh(cls, Z):
        return np.tanh(Z)

    @classmethod
    def softmax(cls, Z):
        pass

    @classmethod
    def relu_derivative(cls, Z):
        """
        计算ReLU激活函数的导数。
        参数:
            Z: 输入的张量 (numpy array)。
        返回:
            ReLU导数的numpy array，与输入Z形状相同。
        """
        dZ = np.array(Z, copy=True)  # 复制Z到dZ
        dZ[Z <= 0] = 0  # Z中小于等于0的元素的导数为0
        dZ[Z > 0] = 1  # Z中大于0的元素的导数为1
        return dZ

    @classmethod
    def sigmoid_derivative(cls, Z):
        pass

    @classmethod
    def linear_derivative(cls, Z):
        pass

    @classmethod
    def tanh_derivative(cls, Z):
        pass

    # -------------------------------------------------------------------------
    def compute_Z(self, W, X, b):
        Z = np.dot(W, X) + b
        return Z


class Neural_Network:
    def __init__(self, X, y, layers, alpha=1e-4):
        self.X = X
        self.y = y
        self.layers = layers
        self.alpha = alpha
        pass

    def fit(self, epochs=30000):
        for epoch in range(epochs):
            if epoch % max(1, math.ceil(epochs / 10)) == 0:
                print(f"Iteration {epoch:4d}")
            for layer in self.layers:
                if layer == self.layers[0]:
                    A = layer.forward(self.X)
                else:
                    A = layer.forward(A)
            for layer in self.layers[::-1]:  # loss是BSE,这里直接算出了BCE的导数,后面可以修改为参数
                if layer == self.layers[-1]:  # 注意这里是反向传播,先处理最后一层
                    dZ = layer.A - self.y
                    layer.dZn = dZ
                    dW = np.dot(dZ, layer.A_in.T)  # 应该乘以前一层    一个想法是将输入的a也保存
                    layer.dWn = dW
                    db = np.sum(dZ, axis=1, keepdims=True)
                    layer.dbn = db
                else:
                    dZ, dW = layer.backward(dZ, dW)
                layer.Wn -= self.alpha * layer.dWn
                layer.bn -= self.alpha * layer.dbn

    def predict(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def bce_loss(self):
        pass

    def mse_loss(self):
        pass

    def bce_derivative(self):
        pass

    def mse_derivative(self):
        pass


if __name__ == '__main__':
    # ------------------------数据集-----------------------------
    X_data, y_data = generate_logistic_regression_data(samples=500)
    X_data = X_data.T
    X_train = X_data[:, 0:350]  # 训练集
    y_train = y_data[:, 0:350]
    y_data = y_data.reshape(1, -1)
    X_test = X_data[:, 351:501]  # 测试集
    y_test = y_data[:, 351:501]
    L1 = Dense(8, Dense.relu)
    L2 = Dense(1, Dense.sigmoid)
    Net = Neural_Network(X_train, y_train, [L1, L2])
    Net.fit()
    Y_predict = Net.predict(X_test)
    Y_predict = (Y_predict > 0.5).astype(int)
    # ------------------------------------绘图--------------------------------------
    y_train = y_train.reshape(350, )
    Y_predict = Y_predict.reshape(149, )
    plt.scatter(X_train[0][y_train == 1], X_train[1][y_train == 1], color="blue", label='Class 1')
    plt.scatter(X_train[0][y_train == 0], X_train[1][y_train == 0], color="yellow", label='Class 0')
    plt.scatter(X_test[0][Y_predict == 1], X_test[1][Y_predict == 1], color="red", label='Predict Class 1')
    plt.scatter(X_test[0][Y_predict == 0], X_test[1][Y_predict == 0], color="green", label='Predict Class 0')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
