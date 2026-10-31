import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

import mlflow

data = load_breast_cancer()

X =pd.DataFrame( data.data, columns = data.feature_names )
y = pd.Series( data.target , name = 'target' )

Xtrain, Xtest, ytrain, ytest = train_test_split( X, y, test_size =0.2, random_state = 42 )


# Grid Search for Hyperparameter Tuning without MLflow  
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20]
}

# grid_search = GridSearchCV(
#     estimator = RandomForestClassifier( random_state = 42 ),
#     param_grid = param_grid,
#     cv = 5,
#     scoring = 'accuracy',
#     n_jobs = -1,
#     verbose = 1
# )

# grid_search.fit( Xtrain, ytrain )

# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# best_model = grid_search.best_estimator_


# print("---"*45)
# print(f"Best Parameters: {best_params}")
# print(f"Best Cross-Validation Score: {best_score}")
# print(f"Best Model: {best_model}")
# print(f"Test Set Score: {best_model.score(Xtest, ytest)}")
# print("---"*45)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Hyperparameter Tuning with MLflow Tracking
mlflow.set_experiment("Breast_Cancer_Classification_Experiment")
with mlflow.start_run( run_name = "Random_Forest_Grid_Search" ) as run:
    grid_search = GridSearchCV(
        estimator =RandomForestClassifier( random_state = 42 ),
        param_grid = param_grid,
        cv = 5,
        scoring = 'accuracy',
        n_jobs = -1,
        verbose = 1
    )
    
    grid_search.fit( Xtrain, ytrain )
    
    for i in range( len(grid_search.cv_results_['params']) ):
        with mlflow.start_run( run_name = f"Grid_Search_Run_{i}", nested = True ) as nested_run:
            mlflow.log_params( grid_search.cv_results_['params'][i] )
            mlflow.log_metric( 'mean_test_score', grid_search.cv_results_['mean_test_score'][i] )
            
            
        
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", best_score)
    
    train_df = Xtrain.copy()
    train_df['target'] = ytrain
    
    train_df = mlflow.data.from_pandas( train_df )
    mlflow.log_input( train_df, "train_data" )
    
    test_df = Xtest.copy()
    test_df['target'] = ytest
    
    test_df = mlflow.data.from_pandas( test_df )
    mlflow.log_input( test_df, 'test_data' )
    
    
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(best_model, "best_model", skops_trusted_types=["sklearn.tree._tree.Tree"] )
    
    
    print("---"*45)
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score}")
    print(f"Best Model: {best_model}")
    print(f"Test Set Score: {best_model.score(Xtest, ytest)}")
    print("---"*45)
