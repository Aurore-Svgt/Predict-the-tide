from sklearn.model_selection import train_test_split
from loguru import logger
import numpy as np
import pandas as pd
import os

surge1_columns = [
    'surge1_t0', 'surge1_t1', 'surge1_t2', 'surge1_t3', 'surge1_t4',
    'surge1_t5', 'surge1_t6', 'surge1_t7', 'surge1_t8', 'surge1_t9' ]
surge2_columns = [
    'surge2_t0', 'surge2_t1', 'surge2_t2', 'surge2_t3', 'surge2_t4',
    'surge2_t5', 'surge2_t6', 'surge2_t7', 'surge2_t8', 'surge2_t9' ]

def test(models, path="../data/", ratio=0.2, train_score=False, prediction=False):
    '''
    module to test a given model for prediction
    the model should conform to sklearn nomenclature (fit, predict methods)
    name: string with name of model

    '''

    X_train = np.load(path+'X_train_surge.npz')
    Y_train = pd.read_csv(path+'Y_train_surge.csv')

    train, test = train_test_split(np.linspace(0, len(Y_train)-1, num=len(Y_train), dtype = int), test_size=ratio)

    logger.info(f"Length of train set {len(train)}")
    logger.info(f"Length of test set {len(test)}")

    slp_train, slp_eval = X_train["slp"][train,:,:,:], X_train["slp"][test,:,:,:]
    surge1_train, surge1_eval = X_train["surge1_input"][train,:], X_train["surge1_input"][test,:]
    surge2_train, surge2_eval = X_train["surge2_input"][train,:], X_train["surge2_input"][test,:]
    Y_train, Y_eval = Y_train.loc[train].drop('id_sequence', axis=1), Y_train.loc[test].drop('id_sequence', axis=1)

    for name, model in models.items():
        logger.info(f"Fitting model {name}")
        model.fit(slp_train, surge1_train, surge2_train, Y_train)
        logger.info(f"Model {name} fitted")
        y_pred1, y_pred2 = model.predict(slp_eval, surge1_eval, surge2_eval)
        logger.info("Predicted test values")

        if train_score:
            y_pred1_train, y_pred2_train = model.predict(slp_train, surge1_train, surge2_train)
            logger.info(f"{name}: Predicted train values")
            y_pred_train = np.append(y_pred1_train, y_pred2_train, axis=1)
            y_pred_train = pd.DataFrame(y_pred_train, columns= surge1_columns +surge2_columns)
            train_score = surge_prediction_metric(Y_train, y_pred_train)
            logger.info(f"{name}: Train score: {train_score}")


        y_pred = np.append(y_pred1, y_pred2, axis=1)
        y_pred = pd.DataFrame(y_pred, columns= surge1_columns +surge2_columns)
        test_score =  surge_prediction_metric(Y_eval, y_pred)
        logger.info(f"{name}: Test score: {test_score}")

        if prediction:
            predict(model, name)

def surge_prediction_metric(dataframe_y_true, dataframe_y_pred):
    weights = np.linspace(1, 0.1, 10)[np.newaxis]

    surge1_score = (weights * (dataframe_y_true[surge1_columns].values - dataframe_y_pred[surge1_columns].values)**2).mean()
    surge2_score = (weights * (dataframe_y_true[surge2_columns].values - dataframe_y_pred[surge2_columns].values)**2).mean()

    return surge1_score + surge2_score

def predict(model, name:str, path:str="../data/", results:str="./predictions"):

    X_train = np.load(path+'X_train_surge.npz')
    Y_train = pd.read_csv(path+'Y_train_surge.csv')
    X_test = np.load(path+'X_test_surge_178mikI.npz')

    logger.info(f"Fitting model {name}")
    model.fit(X_train["slp"], X_train["surge1_input"], X_train["surge2_input"], Y_train.drop('id_sequence', axis=1))
    logger.info(f"Model {name} fitted")
    y_pred1, y_pred2 = model.predict(X_test["slp"].reshape(-1, 40, 41,41), X_test["surge1_input"], X_test["surge2_input"])
    logger.info("Predicted test values")
    y_pred = np.append(y_pred1, y_pred2, axis=1)
    y_pred = pd.DataFrame(y_pred, columns= surge1_columns +surge2_columns)
    y_pred["id_sequence"] = X_test["id_sequence"]
    y_pred.set_index("id_sequence", inplace=True)
    save_path = results+"/prediction_"+name+".csv"
    y_pred.to_csv(save_path)
    logger.info(f"Saved results to {save_path}")
