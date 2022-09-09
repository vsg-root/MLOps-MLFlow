import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
import mlflow
import mlflow.tensorflow


(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_val = x_val.reshape((len(x_val),np.prod(x_val.shape[1:])))

x_train = x_train.astype("float32")
x_val = x_val.astype("float32")

x_train /= 255
x_val /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_val = np.utils.to_categorical(y_val, 10)

def train_dl(n_hidden_layers, n_units, activation, drop_out, epochs):
    mlflow.search_experiments("DLExperiment")


    with mlflow.start_run():
        mlflow.tensorflow.autolog()

        # Tags Register
        mlflow.set_tag("n_hidden_layers",n_hidden_layers)
        mlflow.set_tag("n_units",n_units)
        mlflow.set_tag("activation",activation)
        mlflow.set_tag("drop_out",drop_out)
        mlflow.set_tag("epochs",epochs)


        model = Sequential()
        model.add(Dense(units=n_units, activation=activation, input_dim=784))
        model.add(Dropout(drop_out))
        for n in range(n_hidden_layers):
            model.add(Dense(units=n_units, activation=activation, input_dim=784))
            model.add(Dropout(drop_out))
        model.add(Dense(units=10, activation='softmax'))

        model.compile(optimazer='adam', loss="categorical_crossentropy", metric=["accuracy"])
        model.summary()

        History = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))


        # Plot and Save Graphics
        History.history.keys()
        loss = plt.plot(History.history["val_loss"])
        plt.savefig("loss.png")
        accuracy = plt.plt(History.history["val_accuracy"])
        plt.savefig("Accuracy.png")

         # Model
        mlflow.sklearn.log_model(model, "ModelDL")

        # Run Information
        print("Model: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()


n_hidden_layers = [1, 2, 3]
n_units = [16, 32, 64]
activation = ["relu", "tahn"]
drop_out = [0.1, 0.2]
epochs = [5, 10, 20]


for layers in n_hidden_layers:
    for unity in n_units:
        for func_activation in activation:
            for drop in drop_out:
                for epoch in epochs:
                    train_dl(layers, unity, func_activation, drop, epoch) 