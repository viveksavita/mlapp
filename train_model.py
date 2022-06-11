# Script to train machine learning model.
from ml.data import process_data
from ml.model import train_model , inference, compute_model_metrics
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Add the necessary imports for the starter code.

# Add code to load in the data.


df = pd.read_csv("./data/census.csv")

#print(df.columns) 
#print(df[" salary"])

df = df.rename(columns={"age":"age", 
" workclass":"workclass",
" fnlgt":"fnlgt",
" education":'education', 
' education-num': 'education-num',
 ' marital-status':'marital-status', 
 ' occupation' :'occupation', 
 ' relationship': 'relationship', 
 ' race':'race', 
 ' sex' :'sex',
  ' capital-gain':'capital-gain', 
  ' capital-loss':'capital-loss', 
 ' hours-per-week':'hours-per-week', 
  ' native-country':'native-country',
       ' salary': 'salary'})
print(df.columns) 

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

model = train_model(X_train,y_train)

pickle.dump(model, open("./model/logisticRegression.sav", 'wb'))

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)


# Train and save a model.
