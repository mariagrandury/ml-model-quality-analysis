import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers


def download_data(url, cache_dir):
    file_name = url.split('/')[-1]
    dir_name = file_name.split('.')[0]
    tf.keras.utils.get_file(file_name, origin=url, extract=True, cache_dir=cache_dir)
    data_dir = os.path.join('datasets/' + dir_name)
    return data_dir


def load_dataset_from_directory(data_dir, split, size, batch_size, shuffle, subset):
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
    tfds.load(dataset, split=split, as_supervised=as_supervised, shuffle_files=shuffle_files)


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(180, 180, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)