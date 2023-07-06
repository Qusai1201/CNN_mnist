import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def create_model(conv_size, pool_size, kernel_size, activation, dropout_rate):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(conv_size[0], kernel_size=kernel_size, activation=activation),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Conv2D(conv_size[1], kernel_size=kernel_size, activation=activation),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Flatten(),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

batch_size = 64
epochs = 5

param_grid = {
    'conv_size': [(64, 128), (128, 256)],
    'activation': ['relu', 'LeakyReLU'],
    'pool_size': [(2, 2), (3, 3)],
    'kernel_size': [(3, 3), (2, 2)],
    'dropout_rate': [0.3, 0.4, 0.5]
}

model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='CNN_mnist.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

grid_results = grid.fit(x_train, y_train, validation_split=0.1, callbacks=[model_checkpoint_callback])

print("best params:", grid_results.best_params_)

best_model = grid_results.best_estimator_.model

best_model.fit(x_train, y_train, batch_size=batch_size, epochs=20, validation_split=0.1, callbacks=[model_checkpoint_callback])

test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
