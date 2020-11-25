"""Helper functions to load and preprocess data."""


import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers


def download_data(url, cache_dir):
    """Download data.

    :param url: url to the zip or tar file to download the data
    :param cache_dir: directory to store the downloaded data
    """
    file_name = url.split('/')[-1]
    dir_name = file_name.split('.')[0]
    tf.keras.utils.get_file(file_name, origin=url, extract=True, cache_dir=cache_dir)
    data_dir = os.path.join('datasets/' + dir_name)
    return data_dir


def load_dataset_from_directory(data_dir, split, size, batch_size, shuffle, subset):
    """Load the dataset from a directory. Could ask for the data type, now it is assumed to be 'image'."""
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=split,
        subset=subset,
        seed=123,
        image_size=size,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataset


def load_tensorflow_dataset(dataset, split, as_supervised=True, shuffle_files=True):
    """Load the dataset as a TensorFlow dataset."""
    tfds.load(dataset, split=split, as_supervised=as_supervised, shuffle_files=shuffle_files)


def data_augmentation_layer(size):
    """Create a data augmentation layer to train image models.

    :param size: images height and width
    """
    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(size, size, 3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
      ]
    )
    return data_augmentation