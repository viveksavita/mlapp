

import pytest
import numpy
import pandas as pd
from ml.data import process_data
from fastapi.testclient import TestClient
from ml.model import train_model , inference, compute_model_metrics
import sklearn
import pickle
import os
import json

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

@pytest.fixture
def data():
    df = pd.read_csv(os.path.join(os.getcwd(),"data/cleaned_data.csv"))

    return df

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == { "greeting": "Welcome!"}

def test_post_predict():
    r = client.post("/predict")
    assert r.status_code != 200




def test_inference_post_1():
    data = {
    "age": 32,
    "workclass": "Private",
    "fnlgt": 338409,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 38,
    "native_country": "Cube"
  }
    r = client.post("/predict", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary": " <=50K"}


def test_inference_post_2():
    data = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 191765,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Other-relative",
    "race": "Black",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 2339,
    "hours_per_week": 40,
    "native_country": "Trinadad&Tobago"
  }
    r = client.post("/predict", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary": " >50K"}



  


def test_process_data(data):
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]

    features, label, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True)

    assert isinstance(features, numpy.ndarray)
    assert isinstance(label, numpy.ndarray)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)
    
def test_inference(data):
    model = pickle.load(open(os.path.join(os.getcwd(),"model/logisticRegression.sav"), 'rb'))
    encoder = pickle.load(open(os.path.join(os.getcwd(),"model/encoder"), 'rb'))
    lb = pickle.load(open(os.path.join(os.getcwd(),"model/lb"), 'rb'))

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    label = "salary"

    X_test, y_test, encoder, lb = process_data(data, categorical_features=cat_features, 
        label=label, training=False, encoder=encoder, lb=lb)

    preds = inference(model, X_test)
    assert len(preds) == len(y_test)



def test_compute_model_metrics(data):
    model = pickle.load(open(os.path.join(os.getcwd(),"model/logisticRegression.sav"), 'rb'))
    encoder = pickle.load(open(os.path.join(os.getcwd(),"model/encoder"), 'rb'))
    lb = pickle.load(open(os.path.join(os.getcwd(),"model/lb"), 'rb'))

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    label = "salary"

    X_test, y_test, encoder, lb = process_data(data, categorical_features=cat_features, 
        label=label, training=False, encoder=encoder, lb=lb)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert  0.0 <= precision  
    assert precision <=1.0




