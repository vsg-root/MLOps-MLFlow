from cgi import test
from re import X
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

SEED = 112
x_train, x_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.3, 
                                                  random_state=SEED)

mlflow.set_experiment("test_mlflow0")
with mlflow.start_run():

    naive_bayes = GaussianNB()
    naive_bayes.fit(x_train, y_train)
    y_pred = naive_bayes.predict(x_val)

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
    confusion = plot_confusion_matrix(naive_bayes, x_val, y_val)
    plt.savefig("Confusion.png")
    roc = plot_roc_curve(naive_bayes, x_val, y_val)
    plt.savefig("roc.png")

    # Registry Graphics
    mlflow.log_artifact("Confusion.png")
    mlflow.log_artifact("roc.png")
    
    # Modelo
    mlflow.sklearn.log_model(naive_bayes, "ModelNB")

    # Run Information
    print("Modelo: ", mlflow.active_run().info.run_uuid)

mlflow.end_run()