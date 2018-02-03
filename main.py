
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import itertools
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier    
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib 
from sklearn import preprocessing

def main():   
    
# COLUMN NAMES
# 1. preg_count    
# 2. glucose_concentration    
# 3. blood_pressure    
# 4. skin_thickness    
# 5. serum_insulin    
# 6. bmi   
# 7. pedigree_function   
# 8. age   
# 9. class: 1 - yes diabetes, 0 - no diabetes
    
#     1. LOAD DIABETES DATA INTO PANDAS DATAFRAME
    df_diabetes = pd.read_csv(filepath_or_buffer="diabetes.csv")
    
    df_diabetes['preg_count'] = df_diabetes['preg_count'].map( lambda x : df_diabetes.preg_count.median() if x == 0 else x)
    df_diabetes['glucose_concentration'] = df_diabetes['glucose_concentration'].map( lambda x : df_diabetes.glucose_concentration.median() if x == 0 else x)
    df_diabetes['blood_pressure'] = df_diabetes['blood_pressure'].map( lambda x : df_diabetes.blood_pressure.median() if x == 0 else x)
    df_diabetes['skin_thickness'] = df_diabetes['skin_thickness'].map( lambda x : df_diabetes.skin_thickness.median() if x == 0 else x)
    df_diabetes['serum_insulin'] = df_diabetes['serum_insulin'].map( lambda x : df_diabetes.serum_insulin.median() if x == 0 else x)
    df_diabetes['bmi'] = df_diabetes['bmi'].map( lambda x : df_diabetes.bmi.median() if x == 0 else x)

    result = df_diabetes.apply(lambda x: sum(x==0), axis=0) 
#     print("ZERO VALUES (0) BY COLUMNS:")
#     print(result)
#     print()
#     
    result = df_diabetes.apply(lambda x: sum(x.isnull()), axis=0) 
#     print("MISSING VALUES (NAN) BY COLUMNS:")
#     print(result)
#     print()

#     PRINT DATAFRAME INFORMATION
#     df_diabetes.info()    
#     print()

#     2. DEFINE THE FEATURES 
    X = df_diabetes.drop(labels="class", axis=1)
    
#     X = preprocessing.normalize(X)
    
#     3. DEFINE THE TARGET
    y = df_diabetes["class"]
    y_unique_class = list(y.unique())
    
#     4. GET TRAIN AND TEST DATA 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
#     5. SCALE THE DATA - STANDARDIZE FEATURES BY REMOVING THE MEAN AND SCALING TO UNIT VARIANCE
    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
#     6. CREATE MULTI-LAYER PERCEPTRON CLASSIFIER MODEL
    mlp_model = MLPClassifier(activation="relu", hidden_layer_sizes=(2,2,2), max_iter=5000, random_state=1)
# 
# #     mlp_model = MLPClassifier(activation="identity", hidden_layer_sizes=(10, 10, 10, 10, 10), max_iter=500, random_state=1)
# 
#     mlp_model = MLPClassifier(activation="identity", hidden_layer_sizes=(5, 5, 5, 5, 5), max_iter=1500, random_state=1)
#     
#     mlp_model = MLPClassifier(activation="identity", hidden_layer_sizes=(5, 5, 5, 5, 5), solver="lbfgs", max_iter=1000, random_state=1)
#     
#     mlp_model = MLPClassifier(activation="tanh", hidden_layer_sizes=(5, 5, 5, 5, 5), solver="adam", max_iter=1000, random_state=1)
    
#     7. TRAIN THE MODEL WITH TRAIN DATA 
    mlp_model.fit(X_train, y_train)
    
#     8. GET TARGET PREDICTED VALUE
    y_predicted = mlp_model.predict(X_test)      
   
#     9. MODEL EVALUATION FOR DATA CLASSIFICATION
#     ACCURACY SCORE
    accuracy_score_value = accuracy_score(y_test, y_predicted) * 100
    accuracy_score_value = float("{0:.2f}".format(accuracy_score_value))    
    print("Accuracy Score: {} %".format(accuracy_score_value))
    print()
    
#     CONFUSION MATRIX
    confusion_matrix_result = confusion_matrix(y_test, y_predicted)
    
#     SHOW CONFUSION MATRIX PLOT
    plot_confusion_matrix(confusion_matrix_result, 
                          y_class=y_unique_class, 
                          plot_title="Confusion Matrix - Diabetes Data", 
                          plot_y_label="Test Diabetes Class", 
                          plot_x_label="Predicted Diabetes Class")      
    print(confusion_matrix_result)
    print()    
    
#     10. SAVE MODEL TO A .PKL FILE    
    joblib.dump(mlp_model, "mlp_classifier.pkl")
    
#     11. LOAD MODEL FROM .PKL FILE
    mlp_classifier_loaded = joblib.load('mlp_classifier.pkl')
 
#     12. PREDICT DATA SET USING LOADED MODEL
    mlp_classifier_loaded.predict(X_test)

# HOW TO IMPROVE THE MODEL?

#     1. TRY OTHER CLASSIFICATION MODEL FAMILIES (LOGISTIC REGRESSION, RANDOM FOREST, SUPPORT VECTOR MACHINE, ETC.).
#     2. COLLECT MORE DATA IF IT'S CHEAP TO DO SO.
#     3. ENGINEER SMARTER FEATURES AFTER SPENDING MORE TIME ON EXPLORATORY ANALYSIS.
#     4. SPEAK TO A DOMAIN EXPERT TO GET MORE CONTEXT INFORMATION

def plot_confusion_matrix(confusion_matrix, y_class, plot_title, plot_y_label, plot_x_label, normalize=False):
    """
    plot the confusion matrix.
    :param confusion_matrix: confusion matrix value
    :param y_class: target unique class name
    :param plot_title: plot title
    :param plot_y_label: plot y label
    :param plot_x_label: plot x label
    :param normalize: default to false
    :return: None
    """
    try:
        plt.figure()
        plt.imshow(confusion_matrix, interpolation='nearest', cmap="Blues")
        plt.title(plot_title)
        plt.colorbar()
        tick_marks = np.arange(len(y_class))
        plt.xticks(tick_marks, y_class, rotation=45)
        plt.yticks(tick_marks, y_class)
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized Confusion Matrix")
        else:
            print('Confusion Matrix without Normalization')    
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.ylabel(plot_y_label)
        plt.xlabel(plot_x_label)
        plt.tight_layout()    
        plt.show()
    except Exception as ex:
        print( "An error occurred: {}".format(ex))   
        



if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = str(round(end_time - start_time, 1))
    minutes = str(round((end_time - start_time) / 60, 1))
    print("Program Runtime:")
    print("Seconds: " + seconds + " seconds")
    print("Minutes: " + minutes + " minutes")