import numpy as np
from metrics import r2_score


class LinearRegression():
    def __init__(self):
        self._theta = None
        self.cofficients_ = None
        self.intercept_ = None

    def fit_normal(self,X_train,y_train):
        """通过训练数据 fit模型参数"""
        X_temp = np.hstack([np.ones((X_train.shape[0],1)),X_train])
        self._theta = np.linalg.inv((X_temp.T).dot(X_temp)).dot(X_temp.T).dot(y_train)
        self.cofficients_ = self._theta[1:]
        self.intercept_ = self._theta[:1]
        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用Batch梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self,X_train,y_train,n_iters=5,t0=5,t1=50):

        def dJ_theta_stochastic(x_i, y_i, theta):
            return x_i.T.dot(x_i.dot(theta) - y_i) * 2.

        def learning_rate(t):
            return t0 / (t + t1)

        def stochastic_gradient_descent(X_b, y_train,theta, n_iters):
            iters_of_number = len(X_b)
            for n in range(n_iters):
                # shuffle X和y  （注意X_y之间的顺序有关系！）
                X_y = np.hstack([X_b, y_train.reshape((-1, 1))])
                np.random.shuffle(X_y)
                X_b_new = X_y[:, 0:-1]
                y_b_new = X_y[:, -1]

                # 主要算法 因为X y 是随机序列 所以顺序取出来X y做随机梯度下降就可以
                for i in range(iters_of_number):
                    x_i = X_b_new[i]
                    y_i = y_b_new[i]
                    # 计算梯度
                    grad = dJ_theta_stochastic(x_i, y_i, theta)
                    # 更新theta
                    theta = theta - learning_rate(iters_of_number*n+i) * grad
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = stochastic_gradient_descent(X_b, y_train, initial_theta, n_iters)

        self.intercept_ = self._theta[0]
        self.cofficients_ = self._theta[1:]
        return self

    def predict(self,X):
        """给定X 预测y的值"""
        X = np.hstack([np.ones((X.shape[0],1)),X])
        predict_y = X.dot(self._theta)
        return predict_y
    def __repr__(self):
        return "LinearRegression()"

    def score(self,X_test,y_test):
        """根据给定的测试数据集 计算R Square"""
        res_y = self.predict(X_test)
        return r2_score(y_test,res_y)