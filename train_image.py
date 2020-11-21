import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import tensorflow_datasets as tfds
from data_load_split import download_data, load_dataset_from_directory, preprocess_data, data_augmentation


class TrainImageModel:

    def __init__(self):
        self.model = None
        self.data = None


    def load_data(self, training=True, split=0.2, test_samples=100,
                  size=180, batch_size=32, shuffle=True,
                  data_url=None, data_dir=None, data_tf=None):
        """Load and preprocess data.


        :param training: Boolean, True if the model is not pretrained (default: True)
        :param split: Percentage of samples for validation, if training is True (default: 20)
        :param test_samples: Number of samples to test the model (default: 100)
        :param size: Size to resize the images (default: (256,256))
        :param batch_size: Size of the batches o data (default: 32)
        :param shuffle: Whether to shuffle the data (default: True)
        :param data_url: URL to the zip or tar file to download the data.
        :param data_dir: Path to the directory containing the data.
            This main directory should have subdirectories with the names of the classes.
        :param data_tf: Name of the TensorFlow dataset, check list at tfds.list_builders().
        """
        # Download data from url
        if data_url:
            data_dir = download_data(data_url, cache_dir='./')


        # Load data from directory
        size = (size, size)
        if data_dir:
            if training:
                train_ds = load_dataset_from_directory(data_dir, split, size, batch_size, shuffle, subset='training')
                val_ds = load_dataset_from_directory(data_dir, split, size, batch_size, shuffle, subset='validation')
            else:
                test_ds = load_dataset_from_directory(data_dir, split, size, batch_size, shuffle, subset='validation')


        # Load tensorflow dataset
        if data_tf:
            split = "train[:" + str(test_samples) + "]"
            test_ds = tfds.load(data_tf, split=split, as_supervised=True, shuffle_files=shuffle)


        print('\nImages and labels shapes:')
        ds = train_ds or test_ds
        for image_batch, labels_batch in ds.take(1):
            print(image_batch.shape)
            print(labels_batch.shape)


        # Preprocess data
        if training:
            train_ds = preprocess_data(train_ds)
            val_ds = preprocess_data(val_ds)
            processed_data = train_ds, val_ds
        else:
            processed_data = preprocess_data(test_ds)


        self.data = processed_data
        return processed_data


    def train_model(self, processed_data, size, channels=3, num_classes=5,
                    optimizer='adam', loss=None, metrics='accuracy', epochs=15):
        """Create, compile and fit the model.

        :param processed_data:
        :param size:
        :param channels:
        :param num_classes:
        :param optimizer:
        :param loss:
        :param metrics:
        :param epochs:
        """

        self.model = Sequential([
            data_augmentation,
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(size, size, channels)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        self.model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[metrics])

        train_ds, val_ds = processed_data
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        return history


    def save_model(self, model_name):
        """Save the model.

        :param model_name: Name of the model, the path to the saved model will be './saved_model/model_name'.
        """
        self.model.save('./saved_model/' + model_name)