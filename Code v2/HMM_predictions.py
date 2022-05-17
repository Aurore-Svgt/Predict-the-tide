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


from numba import njit

@njit
def interp_nb(x_vals, x, y):
    return np.interp(x_vals, x, y)

def predict_observations(model,X,k):
    X_ = X.copy()
    y_obs=[]
    for i in range(k):
        states = model.predict(X_)
        transmat_cdf = np.cumsum(model.transmat_, axis=1)
        random_state = check_random_state(model.random_state)
        next_state = (transmat_cdf[states[-1]] > random_state.rand()).argmax()
        next_obs = model._generate_sample_from_state(next_state, random_state)
        y_obs.append(next_obs)
        X_.append(next_obs)
    return X_,y_obs
    


if __name__ == '__main__':
    
    X_test = np.load('X_test_surge_178mikI.npz')
    
    model_1 = pickle.load(open("hmm_model_1.pkl", "rb"))
    model_2 =pickle.load(open("hmm_model_2.pkl", "rb"))

    n = len(X_test['t_slp'])
    time_step = 10816.0

    
    
    final_predictions = pd.DataFrame(columns= ['id_sequence'] + surge1_columns +surge2_columns )
    for i in tqdm(range(125,n)):
        y_pred = [X_test['id_sequence'][i]]
        for id in ['1','2']:
            t_min = max( X_test['t_slp'][i][0], X_test['t_surge'+id+'_input'][i][0])
            t_max = min( X_test['t_slp'][i][-1], X_test['t_surge'+id+'_input'][i][-1])

            time_axis = [t_min+j*time_step for j in range(32)]
            n_time = 32 #len(time_axis)
            
            surge_id=np.array( [interp_nb(time_axis, X_test['t_surge'+id+'_input'][i], X_test['surge'+id+'_input'][i]) ])

            X = np.concatenate( (X_test['slp'][i][:n_time,:],surge_id.T) ,axis = 1 ).tolist()
            k = int( (X_test['t_surge'+id+'_output'][i][-1] - X_test['t_surge'+id+'_output'][i][0] )/time_step )+2

            if id =='1':
                _,y_obs = predict_observations(model_1,X,k)
            else:
                _,y_obs = predict_observations(model_2,X,k)
            
            
            next_surges=[ y[-1] for y in y_obs]
            next_surges = interp_nb(X_test['t_surge'+id+'_output'][i], [time_axis[-1]+j*time_step for j in range(1,k+1)], next_surges).tolist()
            y_pred += next_surges
            # print(surge_id)
            # print(next_surges)
            # plt.plot(surge_id[0])
            # plt.plot(list(surge_id[0])+next_surges)
            # plt.show()
        final_predictions.loc[i]=y_pred
        


        #break
    final_predictions.to_csv("hmm_predictions_2.csv",index=False)

    
