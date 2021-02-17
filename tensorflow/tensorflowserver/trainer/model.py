
"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .preprocessor import preprocess as prep_func
from .postprocessor import postprocess as post_func
import tensorflow as tf


class BaseModel(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def build(self, *args, **kwargs):
        pass

    def train(self, x, y, *args, **kwargs):
        pass

    def evaluate(self, x, y, *args, **kwargs):
        pass

    def predict(self, x, *args, **kwargs):
        pass


def input_dataset_fn(features, labels, shuffle, num_epochs, batch_size):
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


class Model(BaseModel):

    def __init__(self, input_dim=None, learning_rate=None, dirpath=None):

        self.INPUT_DIM = None

        if dirpath:
            self.load(dirpath)
        elif input_dim:
            if learning_rate is None:
                raise Exception(
                    "When 'input_dim' is given, 'learning_rate' must be given too."
                )
            self.build(input_dim, learning_rate)
        else:
            raise Exception("'input_dim' or 'dirpath' must be given.")

    def build(self, input_dim, learning_rate):
        self.INPUT_DIM = input_dim

        Dense = tf.keras.layers.Dense
        model = tf.keras.Sequential(
        [
            Dense(
                100, activation=tf.nn.relu,
                kernel_initializer='uniform',
                input_shape=input_dim,
            ),
            Dense(50, activation=tf.nn.relu),
            Dense(1, activation=tf.nn.sigmoid)
        ])

        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

        # Compile tf.keras model
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.model = model

    def train(self, training_dataset,
              num_train_examples, num_epochs,
              validation_dataset, callbacks=None):

        self.model.fit(
            training_dataset,
            steps_per_epoch=None,
            # steps_per_epoch=int(num_train_examples / args.batch_size),
            epochs=num_epochs,
            validation_data=validation_dataset,
            validation_steps=1,
            verbose=1,
            callbacks=callbacks)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def save(self, dirpath,
             overwrite=True, include_optimizer=True,
             *args, **kwargs):
        self.model.save(dirpath,
                        overwrite=overwrite,
                        include_optimizer=include_optimizer,
                        save_format='tf',
                        *args, **kwargs)

    def load(self, dirpath, *args, **kwargs):
        self.model = tf.keras.models.load_model(
            dirpath, *args, **kwargs
        )
        self.INPUT_DIM = self.model.input_shape[1:]

    def predict(self, inputs):
        return self.model.predict(
            inputs, batch_size=None, verbose=0, 
            steps=None, callbacks=None,
            max_queue_size=10, workers=1,
            use_multiprocessing=False,
        )
