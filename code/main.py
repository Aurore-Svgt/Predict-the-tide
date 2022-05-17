from utils.test_model import *
from multisregressors import *

test(   {
            "LASSO without surge input": MultiLASSORegressor(alpha=0.05, standardize=True),
            "ElasticNet": MultiElasticNetRegressorSurge(alpha=0.05, l1_ratio=0.2),
            "LASSO": MultiLASSORegressorSurge(alpha=0.05, standardize=True),
            "SVR": MultiSVRRegressorSurge(nu=.9, C=1, gamma=1e-5),
        }, ratio=0.2, train_score=True, prediction=False)
