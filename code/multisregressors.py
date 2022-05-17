from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from tqdm import tqdm
from utils.test_model import *
import loguru as logger
import numpy as np
from threading import Thread

class MultiRegressor():
    def __init__(self, n_components=15, standardize=True,**kwargs):
        self.components = n_components
        if standardize:
            self._scaler = StandardScaler()
        self.standardize = standardize
        self._pca = IncrementalPCA(n_components=n_components)
        regressor = LinearRegression
        self._build_regressors(regressor, **kwargs)

    def _build_regressors(self, model, **kwargs):
        self.regressors = np.empty((10,2), dtype = model)
        for i in range(10):
            self.regressors[i, 0] = model(**kwargs)
            self.regressors[i, 1] = model(**kwargs)

    def fit(self, slp, surge1, surge2, Y):
        l, t, w, h = slp.shape

        if self.standardize:
            slp = self._scaler.fit_transform(slp.reshape(l*t,w*h)).reshape(l,t, w,h)
        slp_transformed = self._pca.fit_transform(slp.reshape(l*t, w*h)).reshape(l, t*self.components)

        Y1 = Y[surge1_columns].to_numpy()
        Y2 = Y[surge2_columns].to_numpy()
        def fit_regressor(t, loc, Y):
            self.regressors[t, loc].fit(slp_transformed, Y[:,t])

        for i in range(10):
            t1 = Thread(target=fit_regressor, args=(i, 0, Y1))
            t2 = Thread(target=fit_regressor, args=(i, 1, Y2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    def predict(self, slp, surge1, surge2):
        l, t, w, h = slp.shape
        res1 = np.zeros((l, 10))
        res2 = np.zeros((l, 10))

        if self.standardize:
            slp = self._scaler.transform(slp.reshape(l*t,w*h)).reshape(l,t, w,h)
        slp_transformed = self._pca.fit_transform(slp.reshape(l*t, w*h)).reshape(l, t*self.components)

        def predict_regressor(t, loc, res):
            res[:, t] = self.regressors[t, loc].predict(slp_transformed)

        for i in range(10):
            t1 = Thread(target=predict_regressor, args=(i, 0, res1))
            t2 = Thread(target=predict_regressor, args=(i, 1, res2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        return res1, res2

class MultiLASSORegressor(MultiRegressor):
    def __init__(self, alpha=0.7, n_components=15, standardize=True, **kwargs):

        self.components = n_components
        self._pca = IncrementalPCA(n_components=n_components)
        regressor = Lasso
        kwargs["alpha"]=alpha
        kwargs["max_iter"]=5000
        self._build_regressors(regressor, **kwargs)
        if standardize:
            self._scaler = StandardScaler()
        self.standardize = standardize

class MultiLASSORegressorSurge():
    def __init__(self, alpha=0.1, n_components=20, standardize=True, **kwargs):

        self.components = n_components
        self._pca = IncrementalPCA(n_components=n_components)
        regressor = Lasso
        kwargs["alpha"]=alpha
        kwargs["max_iter"]=5000
        self._build_regressors(regressor, **kwargs)
        if standardize:
            self._scaler_slp = StandardScaler()
            self._scaler_surge1 = StandardScaler()
            self._scaler_surge2 = StandardScaler()
        self.standardize = standardize


    def _build_regressors(self, model, **kwargs):
        self.regressors = np.empty((10,2), dtype = model)
        for i in range(10):
            self.regressors[i, 0] = model(**kwargs)
            self.regressors[i, 1] = model(**kwargs)

    def fit(self, slp, surge1, surge2, Y):
        l, t, w, h = slp.shape

        if self.standardize:
            slp = self._scaler_slp.fit_transform(slp.reshape(l*t,w*h)).reshape(l,t, w,h)
            surge1 = self._scaler_surge2.fit_transform(surge1).reshape(l, 10)
            surge2 = self._scaler_surge2.fit_transform(surge2).reshape(l, 10)

        slp_transformed = self._pca.fit_transform(slp.reshape(l*t, w*h)).reshape(l, t*self.components)
        X1 = np.concatenate((slp_transformed, surge1), axis=1)
        X2 = np.concatenate((slp_transformed, surge2), axis=1)

        Y1 = Y[surge1_columns].to_numpy()
        Y2 = Y[surge2_columns].to_numpy()

        def fit_regressor(t, loc, Y, X):
            self.regressors[t, loc].fit(X, Y[:,t])

        for i in range(10):
            t1 = Thread(target=fit_regressor, args=(i, 0, Y1, X1))
            t2 = Thread(target=fit_regressor, args=(i, 1, Y2, X2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    def predict(self, slp, surge1, surge2):
        l, t, w, h = slp.shape
        res1 = np.zeros((l, 10))
        res2 = np.zeros((l, 10))

        if self.standardize:
            slp = self._scaler_slp.transform(slp.reshape(l*t,w*h)).reshape(l,t, w,h)
            surge1 = self._scaler_surge2.transform(surge1).reshape(l, 10)
            surge2 = self._scaler_surge2.transform(surge2).reshape(l, 10)

        slp_transformed = self._pca.fit_transform(slp.reshape(l*t, w*h)).reshape(l, t*self.components)
        X1 = np.concatenate((slp_transformed, surge1), axis=1)
        X2 = np.concatenate((slp_transformed, surge2), axis=1)

        def predict_regressor(t, loc, res, X):
            res[:, t] = self.regressors[t, loc].predict(X)

        for i in range(10):
            t1 = Thread(target=predict_regressor, args=(i, 0, res1, X1))
            t2 = Thread(target=predict_regressor, args=(i, 1, res2, X2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        return res1, res2

class MultiElasticNetRegressorSurge(MultiLASSORegressorSurge):
    def __init__(self, alpha=1., l1_ratio=0.5, n_components=20, standardize=True, **kwargs):

        self.components = n_components
        self._pca = IncrementalPCA(n_components=n_components)
        kwargs["alpha"]=alpha
        kwargs["l1_ratio"] = l1_ratio
        kwargs["max_iter"]=5000
        self._build_regressors(ElasticNet, **kwargs)
        if standardize:
            self._scaler_slp = StandardScaler()
            self._scaler_surge1 = StandardScaler()
            self._scaler_surge2 = StandardScaler()
        self.standardize = standardize

class MultiSVRRegressorSurge(MultiLASSORegressorSurge):
    def __init__(self, n_components=20, standardize=True, **kwargs):

        self.components = n_components
        self._pca = IncrementalPCA(n_components=n_components)
        self._build_regressors(NuSVR, **kwargs)
        if standardize:
            self._scaler_slp = StandardScaler()
            self._scaler_surge1 = StandardScaler()
            self._scaler_surge2 = StandardScaler()
        self.standardize = standardize

if __name__ == '__main__':

    test(   {
                "C 1": MultiSVRRegressorSurge(nu=.5, C=1, gamma=0.01),
            }, ratio=0.2, train_score=True, prediction=False)
