import numpy as np


class PCA:
    def __init__(self,n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None
    def fit(self,X,eta=0.01,n_iters=1e4):
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def f(w, X):
            return np.sum(X.dot(w) ** 2) / len(X)

        def df_math(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def gradient_ascent_me(df, X, w, eta, n_iters=1e4, epsilon=1e-8):
            # 将w转换为单位向量
            def w_trans(w):
                return w / np.linalg.norm(w)
            iters = 0
            w = w_trans(w)
            while iters < n_iters:
                last_w = w
                gradient = df(w, X)
                w = w + eta * gradient
                w = w_trans(w)
                if (abs(f(w, X) - f(last_w, X))) < epsilon:
                    break
                iters += 1
            return w

        def dmean(X):
            return X - np.mean(X, axis=0)

        X_pca = dmean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            init_w = np.random.random(X_pca.shape[1])
            w = gradient_ascent_me(df_math,X_pca, init_w, eta=eta)
            self.components_[i,:] = w
            #下一个维度上的X_pca
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w    #画图理解！

        return self
    def transform(self,X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self,X_k):
        """将给定的X，反向映射回原来的特征空间"""
        assert X_k.shape[1] == self.components_.shape[0]

        return  X_k.dot(self.components_)
    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components