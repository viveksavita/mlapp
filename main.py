# Put the code for your API here.

from typing import Union

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
import pickle
from ml.data import process_data
from ml.model import train_model , inference, compute_model_metrics
import pandas as pd

# Instantiate the app.
app = FastAPI()

class Input(BaseModel):
    age : int = Field(example=32)
    workclass : str = Field(example="Private")
    fnlgt : int = Field(example=338409)
    education :str = Field(example="Bachelors")
    education_num : int = Field(example=13)
    marital_status : str = Field(example="Married-civ-spouse")
    occupation : str= Field(example="Prof-specialty")
    relationship : str= Field(example="Wife")
    race : str= Field(example="White")
    sex : str= Field(example="Male")
    capital_gain : float= Field(example=0.0)
    capital_loss: float= Field(example=0.0)
    hours_per_week :float= Field(example=38)
    native_country : str= Field(example="Cube")


@app.post("/items")
async def update_item(input: Input):
    
    model = pickle.load(open("./model/logisticRegression.sav", 'rb'))
    encoder = pickle.load(open("./model/encoder", 'rb'))
    lb = pickle.load(open("./model/lb", 'rb'))

    cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
     "native_country"]
    
    
    value_dict= input.__dict__


    X = pd.DataFrame([value_dict], columns=[
    "age", 
    "workclass",
    "fnlgt",
    'education', 
    'education_num',
   'marital_status', 
    'occupation', 
    'relationship', 
    'race', 
    'sex',
    'capital_gain', 
    'capital_loss', 
    'hours_per_week', 
    'native_country'])

   

    X_processed, y, encoder, lb = process_data(X, categorical_features=cat_features, 
     training=False, encoder=encoder, lb=lb)
    
    preds = inference(model, X_processed)



    return  preds[0].tolist()
