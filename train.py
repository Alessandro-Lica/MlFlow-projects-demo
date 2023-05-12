#Url dataset: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
#Importing required packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
import mlflow
random_state = 42

wine = pd.read_csv('C:\\Users\AlessandroLica\\mlflow demo preparazione\\winequality-red.csv')

#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 5.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = random_state)
X_train_orig = X_train.copy()
X_test_orig = X_test.copy()

#provare MinMaxScaler()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def predict_on_test_data(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    #print({'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)})
    return {'accuracy': round(acc,2), 'f1_score': round(f1, 2), 'AUC': round(auc, 2)}

def RF_tuning(X_train, y_train, random_state_RF):
    # define random parameters grid
    n_estimators = [25,50,100,150,200] # number of trees in the random forest
    max_features = [None, 'sqrt', "log2"] # number of features in consideration at every split
    #max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 5, 8] # minimum sample number to split a node
    min_samples_leaf = [1, 2, 3] # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False] # method used to sample data points

    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap
                  }
    
    classifier = RandomForestClassifier(random_state=random_state_RF)
    model_tuning = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, scoring = 'f1',
                   n_iter = 50, cv = 5, verbose=1, random_state=random_state_RF, n_jobs = -1)
    model_tuning.fit(X_train, y_train)

    #print ('Random grid: ', random_grid, '\n')
    
    best_params = model_tuning.best_params_
    model_tuned = model_tuning.best_estimator_
    best_score = model_tuning.best_score_
    
    # print the best parameters
    print ('Best Parameters: ', best_params,)
    print("Best score: ",best_score,)
    return model_tuned,best_params, best_score

def create_experiment_model(experiment_name, run_name, run_metrics, model, library = 'sklearn', run_params=None, tags = None):
    mlflow.set_tracking_uri("http://localhost:5000") #uncomment this line if you want to use any database like sqlite as backend storage for model
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name = run_name):
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        if not library == None:
            if library == 'sklearn':
                mlflow.sklearn.log_model(model, "model")
            elif library == 'catboost':
                mlflow.catboost.log_model(model, "model")
        if not tags == None:
            mlflow.set_tags(tags)
            
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))

experiment_name = "red wine quality classification"
run_name="Random forest 3"
model_tuned, best_params, best_score = RF_tuning(X_train, y_train, random_state+2)
preds = predict_on_test_data(model_tuned, X_test)
run_metrics = get_metrics(y_test, preds)
print(run_metrics)
tags = {"model": "Random Forest", "model selection": "RandomizedSearchCV", "feature scaling": "Standardization"}
create_experiment_model(experiment_name,run_name,run_metrics,model_tuned,"sklearn",best_params,tags)









