# Put the code for your API here.


import pickle
import pandas as pd
from typing import Union,List
from fastapi import  FastAPI
from ml.data import process_data
from pydantic import BaseModel, Field
from ml.model import inference



# Instantiate the app.
app = FastAPI()


@app.get("/")
async def say_hello():
    return {"greeting": "Welcome!"}

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

@app.post("/predict")
async def update_item(input: List[Input]):
    output = []
    for data in input:
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
        
        
        value_dict= data.__dict__


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
        result = lb.inverse_transform(preds[0])
        output.append(result)

        prediction = pd.DataFrame(output, columns=["salary"])
        prediction = prediction.to_dict('list')

    return  prediction
