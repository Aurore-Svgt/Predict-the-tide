from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from loguru import logger
import numpy as np
from sklearn.utils import check_random_state
import pickle5 as pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
surge1_columns = [
    'surge1_t0', 'surge1_t1', 'surge1_t2', 'surge1_t3', 'surge1_t4',
    'surge1_t5', 'surge1_t6', 'surge1_t7', 'surge1_t8', 'surge1_t9' ]
surge2_columns = [
    'surge2_t0', 'surge2_t1', 'surge2_t2', 'surge2_t3', 'surge2_t4',
    'surge2_t5', 'surge2_t6', 'surge2_t7', 'surge2_t8', 'surge2_t9' ]
    

def make_predictions(new_X_test,model1_path,model2_path):

    
    model_1 = pickle.load(open(model1_path, "rb"))
    model_2 = pickle.load(open(model2_path, "rb"))
    
    y_pred_1 = model_1.predict(new_X_test['1'].reshape(509,32*1682))
    y_pred_2 = model_2.predict(new_X_test['2'].reshape(509,32*1682))

    id_sequence = np.load('X_test_surge_178mikI.npz')['id_sequence']
    y12_pred_train = pd.concat( [ pd.DataFrame(y_pred_1,columns=surge1_columns,index =id_sequence ),
    pd.DataFrame(y_pred_2,columns=surge2_columns,index =id_sequence )],axis=1)

    y12_pred_train.to_csv('predictions_preprocess_0.csv',index_label='id_sequence')

if __name__ == '__main__':
    
    model1_path = "MultiXGB1.pkl"
    model2_path="MultiXGB2.pkl"
    new_X_test = pickle.load(open("new_X_test.pkl", "rb"))
    make_predictions(new_X_test,model1_path,model2_path)
    
    
