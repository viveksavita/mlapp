import requests as rq
import json as js


data1 = {
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

data2 = {
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

def apicallrequest(data):
    result = rq.post('http://127.0.0.1:8000/predict', data=js.dumps(data))
    print(f"==[ result: {result.json()}")
    print(result.status_code)




if __name__=="__main__":
    apicallrequest(data1)
    apicallrequest(data2)
