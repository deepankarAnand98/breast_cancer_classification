from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from data_preprocessing import model_fitting
from data_preprocessing import X_norm,y_norm,X_res,y_res

import pickle

X_train, X_test, y_train, y_test = train_test_split(X_norm,y_norm,test_size=0.2,random_state=10)


# SVM
parameters = {
    "C":[0.1,1,5,10,100,1000],
    "gamma":[0.0001,0.001,0.005,0.1,1,3,5]
}
svc_best_params = model_fitting(X_norm,y_norm,SVC(),parameters,cv=5)

svc_model = SVC(C=svc_best_params["C"],gamma=svc_best_params["gamma"])
svc_model.fit(X_train,y_train)

# Save Model
pkl_filename = "SVM_Classifier.pkl"
with open(pkl_filename,"wb") as f:
    pickle.dump(svc_model,f)

# Random Forest
parameters = {
    'n_estimators':[100,500,1000,5000],
    'max_depth':[7,8,9,10,11]
}

rf_best_params = model_fitting(X_res, y_res, RandomForestClassifier(), parameters, cv=5)
rf_model = RandomForestClassifier(n_estimators=rf_best_params['n_estimators'],max_depth=rf_best_params['max_depth'])
rf_model.fit(X_train,y_train)

# Save Model
pkl_filename = "Random_Forest_Classifier.pkl"
with open(pkl_filename,"wb") as f:
    pickle.dump(rf_model,f)

# XGBoost Classifier

parameters = {
    "n_estimators":[50,100,200,350,500,],
    "max_depth":[6,7,8,9,10,11]
}

xgb_best_params = model_fitting(X_train,y_train, XGBClassifier(),parameters,cv=5)
xgb_model = XGBClassifier(n_estimators=xgb_best_params['n_estimators'],max_depth=xgb_best_params['max_depth'])
xgb_model.fit(X_train,y_train)
pkl_filename = "XGBoost_Classifier.pkl"
with open(pkl_filename,"wb") as f:
    pickle.dump(xgb_model,f)
