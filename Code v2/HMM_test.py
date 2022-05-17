import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle5 as pickle

def preprocess_data(X_train):
    #n = 10
    n = len(X_train['t_slp'])
    

    time_step = 10816.0

    new_X_train= {}
    for id in ['1','2']:
        X = [None for i in range(n)]
        lengths = [None for i in range(n)]
        for i in tqdm(range(n)):
            

            ## interpolate 

            t_min = max( X_train['t_slp'][i][0], X_train['t_surge'+id+'_input'][i][0])
            t_max = min( X_train['t_slp'][i][-1], X_train['t_surge'+id+'_input'][i][-1])

            time_axis = [t_min+j*time_step for j in range(32)]
            # j=0
            # while t_min+j*time_step <=t_max:
            #     time_axis.append(t_min+j*time_step)
            #     j+=1

            n_time = 32 #len(time_axis)
            
            surge_id=np.array( [np.interp(time_axis, X_train['t_surge'+id+'_input'][i], X_train['surge'+id+'_input'][i]) ])

            X[i] = np.concatenate( (X_train['slp'][i][:n_time,:],surge_id.T) ,axis = 1 ).tolist()
            lengths[i] = n_time

        new_X_train[id] = X
    pickle.dump(new_X_train,open( "hmm_X_test.pkl", "wb" ))
    


def main():
    X_test = np.load('X_test_surge_178mikI.npz')
    preprocess_data(X_test)

if __name__=="__main__":
    main()