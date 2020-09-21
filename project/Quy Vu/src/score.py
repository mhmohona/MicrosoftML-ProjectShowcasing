# pylint: disable=global-variable-undefined
"""Scoring script"""
import logging
import json
import os
import pandas as pd
import joblib


def init():
    """One time initialisation"""
    global MODEL
    MODEL = joblib.load(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_model.pkl'))

    logging.info('Model loaded')


def run(input_json):
    """Predict from input data

    Parameters
    ----------
    input_json : str
        input data in json format, with key being feature name

    Returns
    -------
    list
        predictions
    """
    features = pd.DataFrame.from_dict(
        json.loads(input_json),
        orient='index'
    ).T

    prediction = MODEL.predict_proba(features)[:, 0]
    logging.info('Prediction done for %s records', features.shape[0])

    return prediction.tolist()
