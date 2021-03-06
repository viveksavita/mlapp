# Script to train machine learning model.
from ml.data import process_data
from ml.model import train_model , inference, compute_model_metrics
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# Add the necessary imports for the starter code.

# Add code to load in the data.


df = pd.read_csv("./data/census.csv")

df = df.rename(columns={
    "age":"age", 
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
    ' salary': 'salary'
    })
print(df.columns) 

df = df.drop_duplicates()
df.to_csv("./data/cleaned_data.csv", index=False)



# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

#print(test.loc[0])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

with open("./model/encoder", "wb") as f: 
    pickle.dump(encoder ,f)

with open("./model/lb", "wb") as f: 
    pickle.dump(lb, f)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", 
    training=False, 
    encoder=encoder, 
    lb=lb
)


model = train_model(X_train,y_train)

pickle.dump(model, open("./model/logisticRegression.sav", 'wb'))


preds = inference(model, X_test)

p = np.array(np.where(preds>0))
print(p[0][0])
print(test.iloc[p[0][0]])
precision, recall, fbeta = compute_model_metrics(y_test, preds)



def slice_performance(X,cat_features, lab,encode,lb,feature="education"):
    model = pickle.load(open("./model/logisticRegression.sav", 'rb'))
    for cls in X[feature].unique():
        df_temp = X[X[feature] == cls]
        X_test, y_test, encoder, lb = process_data(X, categorical_features=cat_features, 
        label=lab, training=False, encoder=encode, lb=lb)
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)

        with open("slice_output.txt", "a") as f:
            print(f" Score for {cls} : Precision:{precision} , Recall:{recall} ,fbeta:{fbeta}", file=f)


slice_performance(train,cat_features,"salary",encoder,lb,feature="education")
