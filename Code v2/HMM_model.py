from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import check_random_state
import pickle5 as pickle



    


if __name__ == '__main__':
    
    X = pickle.load(open("hmmTrainData/hmm_X_train.pkl","rb" )) 
    lengths = pickle.load(open("hmmTrainData/hmm_lengths_train.pkl", "rb"))

    model_1 = hmm.GMMHMM(n_components=5,n_iter=200,verbose=True).fit( X['1'], lengths['1'])
    model_2 = hmm.GMMHMM(n_components=5,n_iter=200,verbose=True).fit(X['2'], lengths['2'])

    pickle.dump(model_1,open( "GmmHmm_model_1.pkl", "wb" ))
    pickle.dump(model_2,open( "GmmHmm_model_2.pkl", "wb" ))
