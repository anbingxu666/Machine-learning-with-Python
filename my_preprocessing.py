import numpy as np
class my_StandardScaler():
    def __init__(self):
        '''初始化Scaler'''
        self.mean_=None
        self.scale_=None
    def fit(self,X):
        '''传入X，作用：通过X计算参数'''
        assert X.ndim==2,"The X dimesion must be 2-dimension"
        self.mean_ = np.mean(X,axis=0)#np.mean(X[:,i]) for i in range(X.shape[1])
        self.scale_= np.std(X,axis=0)
        return self
    def transform(self,X):
        '''传入矩阵X 返回将X进行均值方差标准化的结果'''
        X_transform = np.empty(shape=X.shape,dtype=float)
        X_transform = (X - self.mean_)/self.scale_ #向量化
        return X_transform

