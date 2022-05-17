from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from loguru import logger
import numpy as np
from sklearn.utils import check_random_state
import pickle5 as pickle
from tqdm import tqdm 

def preprocess_test_data(X_test):
    #n = 10
    n = len(X_test['t_slp'])
    

    time_step = 10816.0

    new_X_test= {}
    for id in ['1','2']:
        X = [None for i in range(n)]
        lengths = [None for i in range(n)]
        for i in tqdm(range(n)):
            

            ## interpolate 

            t_min = max( X_test['t_slp'][i][0], X_test['t_surge'+id+'_input'][i][0])
            t_max = min( X_test['t_slp'][i][-1], X_test['t_surge'+id+'_input'][i][-1])

            time_axis = [t_min+j*time_step for j in range(32)]
            # j=0
            # while t_min+j*time_step <=t_max:
            #     time_axis.append(t_min+j*time_step)
            #     j+=1

            n_time = 32 #len(time_axis)
            
            surge_id=np.array( [np.interp(time_axis, X_test['t_surge'+id+'_input'][i], X_test['surge'+id+'_input'][i]) ])
            v = np.linalg.norm(X_test['slp'][i][:n_time,:])
            X[i] = np.concatenate( (X_test['slp'][i][:n_time,:]/v,surge_id.T) ,axis = 1 ).tolist()

        new_X_test[id] = np.concatenate(X)
    pickle.dump(new_X_test,open( "new_X_test.pkl", "wb" ))


if __name__ == '__main__':

    X_test = np.load('X_test_surge_178mikI.npz')
    preprocess_test_data(X_test)
    
    X = pickle.load(open("hmm_X_train.pkl","rb" ))
    for id in ['1','2']:
        X_ = X[id].reshape(5599,32*1682)
        ## normalize
        for i in range(X_.shape[0]):
            X_[i][:-1] = X_[i][:-1]/np.linalg.norm(X_[i][:-1])
        X['id']=X_
    ## saving
    pickle.dump(X,open( "new_X_train.pkl", "wb" ))