"""Generate a tiny Keras 3 model for smoke-testing the optimization pipeline."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.models import Sequential
import numpy as np


def main():
    rng = np.random.default_rng(42)
    # Small MLP: 16 -> 32 -> 16 -> 10, suitable for Vivado io_parallel
    model = Sequential(
        [
            Dense(32, input_shape=(16,), name='dense_1',
                  kernel_initializer=keras.initializers.RandomNormal(seed=0)),
            ReLU(name='relu_1'),
            Dense(16, name='dense_2',
                  kernel_initializer=keras.initializers.RandomNormal(seed=1)),
            ReLU(name='relu_2'),
            Dense(10, name='dense_3',
                  kernel_initializer=keras.initializers.RandomNormal(seed=2)),
        ],
        name='tiny_mlp',
    )
    model.compile(optimizer='adam', loss='mse')
    # Sanity-check forward pass
    X = rng.standard_normal(size=(4, 16)).astype('float32')
    y = model.predict(X, verbose=0)
    print(f"Model output shape: {y.shape}")

    out_path = '/home/z/my-project/scripts/tiny_mlp.keras'
    model.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
