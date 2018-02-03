
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier    
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib 
from sklearn.model_selection import cross_val_score

def main():
#     read diabetes.csv file
    df_diabetes = pd.read_csv(filepath_or_buffer="diabetes.csv")
     
#      data preprocessing
    df_diabetes['preg_count'] = df_diabetes['preg_count'].map( lambda x : df_diabetes.preg_count.median() if x == 0 else x)
    df_diabetes['glucose_concentration'] = df_diabetes['glucose_concentration'].map( lambda x : df_diabetes.glucose_concentration.median() if x == 0 else x)
    df_diabetes['blood_pressure'] = df_diabetes['blood_pressure'].map( lambda x : df_diabetes.blood_pressure.median() if x == 0 else x)
    df_diabetes['skin_thickness'] = df_diabetes['skin_thickness'].map( lambda x : df_diabetes.skin_thickness.median() if x == 0 else x)
    df_diabetes['serum_insulin'] = df_diabetes['serum_insulin'].map( lambda x : df_diabetes.serum_insulin.median() if x == 0 else x)
    df_diabetes['bmi'] = df_diabetes['bmi'].map( lambda x : df_diabetes.bmi.median() if x == 0 else x)

#     define X features and y label (target)
    X = df_diabetes.drop(labels="class", axis=1)    
    y = df_diabetes["class"]
    
#     data split to select train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
#     Standard Scaler for X features
    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
#     hyperparameter candidates dictonary
    hyperparameter_candidates = [{"hidden_layer_sizes":[(5, 5, 5, 5, 5), (10, 10, 10, 10, 10), (15, 15, 15, 15, 15), (20, 20, 20, 20, 20)], 
                             "max_iter":[500, 1000, 1500, 2000], 
                             "activation":["identity", "logistic", "tanh", "relu"],
                             "solver":["lbfgs", "sgd", "adam"]}]
     
#     initialize gridsearchcv object
    clf = GridSearchCV(estimator=MLPClassifier(), param_grid=hyperparameter_candidates, n_jobs=-1, cv=10)
    
#     train the gridsearchcv object
    clf.fit(X_train, y_train)    
     
    print("Best Score:", clf.best_score_) 
    print()
    print("Best Parameters:") 
    print(clf.best_params_) 
    print()
    
    scores = cross_val_score(clf.best_estimator_, X, y, cv=10)
#     print("Accuracy Score for Train CV: ", results.mean())
    print("Accuracy Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
     
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = str(round(end_time - start_time, 1))
    minutes = str(round((end_time - start_time) / 60, 1))
    print("Program Runtime:")
    print("Seconds: " + seconds + " seconds")
    print("Minutes: " + minutes + " minutes")