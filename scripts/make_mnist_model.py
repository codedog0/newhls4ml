"""Train a small MNIST MLP for the optimization pipeline (kept small so HLS
csynth on a fully-unrolled/low-reuse-factor Dense stack stays tractable)."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    model = Sequential(
        [
            Dense(64, input_shape=(784,), name='dense_1',
                  kernel_initializer=keras.initializers.GlorotUniform(seed=0)),
            ReLU(name='relu_1'),
            Dense(10, name='dense_2',
                  kernel_initializer=keras.initializers.GlorotUniform(seed=1)),
            Activation('softmax', name='softmax'),
        ],
        name='mnist_mlp',
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ]
    model.fit(
        x_train, y_train, epochs=60, batch_size=128, validation_split=0.1,
        callbacks=callbacks, verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy (float32 keras): {acc:.4f}")

    out_path = 'scripts/mnist_mlp.keras'
    model.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
