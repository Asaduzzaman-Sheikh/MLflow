"""
    This file will be used to demonstrate how to create own experiment rather than the default experiment in MLflow.
"""
# Importing Necessary Libraries
import os
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import logging

logger = logging.getLogger('wine_classification')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

log_dir = 'logs'
os.makedirs( log_dir, exist_ok=True )

file_path = os.path.join(log_dir, 'wine_classification.log')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

import warnings
warnings.filterwarnings("ignore")

# Loading the Wine Dataset
wine_data = load_wine()
logger.debug("Wine dataset loaded successfully.")

# Crearting a DataFrame from the Wine Dataset
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df['target']=wine_data.target
logger.debug("DataFrame created from the wine dataset.")

logger.debug(f"DataFrame shape: {df.shape}")    
logger.debug(f"DataFrame columns: {df.columns.tolist()}")
logger.debug(f"DataFrame head:\n{df.head()}")

# Separate features and target variable
X = df.drop('target', axis = 1)
y =df['target']
logger.debug("Features and target variable separated.")
logger.debug(f"Features shape: {X.shape}")
logger.debug(f"Target shape: {y.shape}")

logger.info('Train test split started...')

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
logger.info('Train test split completed.')
logger.debug(f"Training set shape: {Xtrain.shape}, {ytrain.shape}")
logger.debug(f"Testing set shape: {Xtest.shape}, {ytest.shape}")

# Training the Random Forest Classifier
logger.info('Random Forest Classifier training started...')
n_estimators = 200
max_depth = 8

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Creating a new experiment in MLflow
experiment_name = 'Wine_Classification_Experiment'
mlflow.set_experiment(experiment_name)
# Starting MLflow run
with mlflow.start_run() as run:
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    rf_model.fit(Xtrain, ytrain)
    logger.info('Random Forest Classifier training completed.')
    
    logger.info('Model evaluation started...')
    yhat = rf_model.predict(Xtest)
    logger.info('Model evaluation completed.')
    
    logger.info('Calculating metrics...')
    accuracy=accuracy_score(ytest, yhat)
    logger.info(f"Accuracy: {accuracy}")
    
    
    
    logger.info('Classification report:')
    classification_rep = classification_report(ytest, yhat)
    logger.info(f"\n{classification_rep}")
    
    cm=confusion_matrix(ytest, yhat)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    logger.info('Confusion matrix plotted and saved as confusion_matrix.png.')
    
    # Logging parameters, metrics, and model to MLflow
    mlflow.log_params({
        'n_estimators': n_estimators,
        'max_depth': max_depth
    })
    
    mlflow.log_metric('accuracy', accuracy)
    
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__) # Log the current script file
    
    mlflow.set_tags(
        {
        'model_type': 'Random Forest Classifier',
        'dataset': 'Wine dataset',
        'author': 'Asaduzzaman',
        'email': 'asaduzzamansheikh777@gamil.com',
        'project': 'Wine Type Classification'
        }
    )
    
    mlflow.sklearn.log_model(
    rf_model, 
    "random_forest_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    
    logger.info('Parameters, metrics, and artifacts logged to MLflow.')
    logger.info('-----------------------------------------------------')
    
    
    
    


