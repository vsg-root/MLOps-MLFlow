from cgi import test
from re import X
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


credit = pd.read_csv("Credit.csv")

for col in credit.columns:
    if credit[col].dtype == 'object':
        credit[col] = credit[col].astype('category').cat.codes

X = credit.iloc[:, 0:20].values
y = credit.iloc[:, 20].values

SEED = 123
x_train, x_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.3, 
                                                  random_state=SEED)

def train_RF(n_estimators):
    mlflow.set_experiment("test_mlflow_RF")
    with mlflow.start_run():

        Random_Forest = RandomForestClassifier(n_estimators=n_estimators)
        Random_Forest.fit(x_train, y_train)
        y_pred = Random_Forest.predict(x_val)

        # Hyperparameters
        mlflow.log_param("n_estimators", n_estimators)


        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_pred, y_val)
        log = log_loss(y_val, y_pred)

        # Registry
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("log", log)

        # Plot Graphics
        confusion = plot_confusion_matrix(Random_Forest, x_val, y_val)
        plt.savefig("ConfusionRF.png")
        roc = plot_roc_curve(Random_Forest, x_val, y_val)
        plt.savefig("rocRF.png")

        # Registry Graphics
        mlflow.log_artifact("ConfusionRF.png")
        mlflow.log_artifact("rocRF.png")
        
        # Model
        mlflow.sklearn.log_model(Random_Forest, "ModelRF")

        # Run Information
        print("Model: ", mlflow.active_run().info.run_uuid)

    mlflow.end_run()


tree = [50, 100, 500, 750, 1000]
for n in tree:
    train_RF(n)