

from fastapi.testclient import TestClient



# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == { "greeting": "Welcome!"}

def test_post_predict():
    r = client.post("/predict")
    assert r.status_code != 200
   