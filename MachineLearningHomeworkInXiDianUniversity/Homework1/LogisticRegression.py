"""
作者:安炳旭
学号：16130120187


仿照sklearn设计的Logistics模型

准确度采用accuracy
"""
from metrics import accuracy_calculate
import numpy as np

class LogisticRegression():
    def __init__(self):
        self._theta = None
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self,X):
        return 1./(1.+np.exp(-X))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用Batch梯度下降法训练Logistic Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum((y * np.log(y_hat)+(1-y)*np.log(1-y_hat))) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

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

    def predict_probality(self,X):
        X = np.hstack([np.ones((X.shape[0],1)),X])
        return self._sigmoid(X.dot(self._theta))

    def predict(self,X):
        """给定X 预测y的值"""

        temp_y = self.predict_probality(X)
        predict_y = np.array(temp_y>=0.5,dtype="int")
        return predict_y

    def __repr__(self):
        return "LogisticRegression()"

    def score(self,X_test,y_test):
        """根据给定的测试数据集 计算Accuracy score"""
        res_y = self.predict(X_test)
        return accuracy_calculate(y_test,res_y)