
from loguru import logger
import numpy as np
from sklearn.utils import check_random_state
import pickle5 as pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.model_selection import train_test_split
#from catboost import CatBoostRegressor
from xgboost import XGBRegressor
#from xgboost.sklearn import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
#from keras.models import Sequential
#from keras.layers import Dense

surge1_columns = [
    'surge1_t0', 'surge1_t1', 'surge1_t2', 'surge1_t3', 'surge1_t4',
    'surge1_t5', 'surge1_t6', 'surge1_t7', 'surge1_t8', 'surge1_t9' ]
surge2_columns = [
    'surge2_t0', 'surge2_t1', 'surge2_t2', 'surge2_t3', 'surge2_t4',
    'surge2_t5', 'surge2_t6', 'surge2_t7', 'surge2_t8', 'surge2_t9' ]

"""# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss='mae', optimizer='adam')
	return model"""

def surge_prediction_metric(dataframe_y_true, dataframe_y_pred):
    weights = np.linspace(1, 0.1, 10)[np.newaxis]

    surge1_score = (weights * (dataframe_y_true[surge1_columns].values - dataframe_y_pred[surge1_columns].values)**2).mean()
    surge2_score = (weights * (dataframe_y_true[surge2_columns].values - dataframe_y_pred[surge2_columns].values)**2).mean()

    return surge1_score + surge2_score

if __name__ == '__main__':
    
    X = pickle.load(open("hmmTrainData/hmm_X_train.pkl","rb" ))
    y = pd.read_csv('Y_train_surge.csv')
    #t_slp=np.load(r'X_train_surge.npz')['t_slp'].flatten()
    #print(t_slp.shape)
    y_pred_train={}
    y_pred_test={}
    X_train,X_test,y_train,y_test = {},{},{},{}
    for id in ['1','2']:
        X_ = X[id].reshape(5599,32*1682)
        ## normalize
        for i in range(X_.shape[0]):
            X_[i][:-1] = X_[i][:-1]/np.linalg.norm(X_[i][:-1])
        #print(X_.shape)
        if id =='1':
            X_train[id],X_test[id],y_train[id],y_test[id] = train_test_split(X_,y[surge1_columns],test_size=0.15,random_state=42)
        else:
            X_train[id],X_test[id],y_train[id],y_test[id] = train_test_split(X_,y[surge2_columns],test_size=0.15,random_state=42)
        
        model  = Pipeline([('pca', PCA(n_components=10)),
         ('XGB', MultiOutputRegressor( XGBRegressor(reg_lambda=0.1)) )])
        model.fit(X_train[id],y_train[id])
        ## save model
        pickle.dump(model,open( "MultiXGB"+id+".pkl", "wb" ))

        y_pred_train[id] = model.predict(X_train[id])
        print(f" Train score: {mean_squared_error( y_pred_train[id],y_train[id])}")
        y_pred_test[id] = model.predict(X_test[id])
        print(f" Test score: {mean_squared_error(y_pred_test[id],y_test[id])}")

    
    y12_train = pd.concat( [ pd.DataFrame(y_train['1'],columns=surge1_columns),pd.DataFrame(y_train['2'],columns=surge2_columns)],axis=1)
    y12_pred_train = pd.concat( [ pd.DataFrame(y_pred_train['1'],columns=surge1_columns),pd.DataFrame(y_pred_train['2'],columns=surge2_columns)],axis=1)
    print( f" Metric Train value: {surge_prediction_metric(y12_pred_train,y12_train)}" )

    y12_test = pd.concat( [pd.DataFrame(y_test['1'],columns=surge1_columns),pd.DataFrame(y_test['2'],columns=surge2_columns)],axis=1)
    y12_pred_test = pd.concat( [pd.DataFrame(y_pred_test['1'],columns=surge1_columns),pd.DataFrame(y_pred_test['2'],columns=surge2_columns)],axis=1)
    print( f" Metric Test value: {surge_prediction_metric(y12_pred_test,y12_test)}" )
    



    
