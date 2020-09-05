# Importing Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Loading Data
data = pd.read_csv('./data/data.csv')
data.drop(["Unnamed: 32","id"], axis=1, inplace=True)
X = data.drop('diagnosis',axis=1)
y = data['diagnosis']
SM = SMOTE(random_state = 10)
X_res, y_res = SM.fit_resample(X,y)

data_norm = (X_res-X_res.mean())/(X_res.max()-X_res.min())
data_norm = pd.concat([data_norm,y_res],axis=1)

X_norm = data_norm.drop('diagnosis',axis=1)
y_norm = data_norm['diagnosis']

le = LabelEncoder()
le.fit(y_norm)

y_norm = le.transform(y_norm)

def model_fitting(X, y, algorithm, gridsearchparameter, cv):
    np.random.seed(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    grid = GridSearchCV(estimator=algorithm,
                        param_grid=gridsearchparameter,
                        cv=cv,
                        scoring='accuracy',
                        verbose=1,
                        n_jobs=-1)

    results = grid.fit(X_train, y_train)
    best_parameters = results.best_params_
    y_predictions = results.predict(X_test)
    cm = confusion_matrix(y_test, y_predictions)

    print("Best Parameters : \n", best_parameters)
    print("Model Summary : \n", classification_report(y_test, y_predictions))
    print("Accuracy Score : \n", accuracy_score(y_test, y_predictions))
    print(f"Confusion Matrix : \n", cm)
    return best_parameters
