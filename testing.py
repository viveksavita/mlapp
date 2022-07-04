import requests as rq
import json as js

def test():

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
    result = rq.post('http://127.0.0.1:8000/predict', data=js.dumps(data))
    print(f"==[ result: {result.json()}")

if __name__=="__main__":
    test()