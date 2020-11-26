import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from data import download_data, data_augmentation_layer


class TrainImageModel:
    """Class to train image classification models.

    Attributes:
    -----------
    data_type: data type, in this case 'image'
    task: machine learning task, in this case 'classification'
    model: machine learning model
    data: data to test the model with

    Methods:
    --------
    load_data(): load and preprocess data
    train_model(): create, compile and fit the model
    save(): save the model
    """

    def __init__(self, data_type='image', task='classification'):
        self.data_type = data_type
        self.task = task
        self.model = None
        self.data = None


    def load_data(self, training=True, split=0.2, test_samples=100,
                  size=180, batch_size=32, shuffle=True,
                  data_url=None, data_dir=None, data_tf=None):
        """Load and preprocess data.


        :param training: whether to train the model (default: True)
        :param split: percentage of samples for validation, if training is True (default: 20)
        :param test_samples: number of samples to test the model (default: 100)
        :param size: size to resize the images (default: (256,256))
        :param batch_size: size of the batches of data (default: 32)
        :param shuffle: whether to shuffle the data (default: True)
        :param data_url: url to the zip or tar file to download the data
        :param data_dir: path to the directory containing the data
            This main directory should have subdirectories with the names of the classes
        :param data_tf: name of the TensorFlow dataset, check list at tfds.list_builders()
        """
        # Reproducibility
        seed = 123

        # Download data from url
        if data_url:
            data_dir = download_data(data_url, cache_dir='./')

        # Load data from directory
        size = (size, size)
        if data_dir:
            if training:
                train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    data_dir, validation_split=split, subset='training', seed=seed,
                    image_size=size, batch_size=batch_size, shuffle=shuffle
                )
                val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    data_dir, validation_split=split, subset='validation', seed=seed,
                    image_size=size, batch_size=batch_size, shuffle=shuffle
                )

            else:
                # if test_samples: split = test_samples / total_samples
                test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    data_dir, validation_split=split, subset='validation', seed=seed,
                    image_size=size, batch_size=1, shuffle=shuffle
                )

        # Load tensorflow dataset
        if data_tf:
            split = "train[:" + str(test_samples) + "]"
            test_ds = tfds.load(data_tf, split=split, as_supervised=True, shuffle_files=shuffle)
            test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))


        print('\nImages and labels shapes:')
        ds = train_ds or test_ds
        for image_batch, labels_batch in ds.take(1):
            print(image_batch.shape)
            print(labels_batch.shape)


        # Preprocess data
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if training:
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            processed_data = train_ds, val_ds
        else:
            processed_data = test_ds.cache().batch(1).prefetch(buffer_size=AUTOTUNE)


        self.data = processed_data
        return processed_data


    def train_model(self, processed_data, size, channels=3, num_classes=5,
                    optimizer='adam', loss=None, metrics='accuracy', epochs=15):
        """Create, compile and fit the model.

        :param processed_data: tuple (train dataset, validation dataset)
        :param size: images height and width
        :param channels: number of channels
        :param num_classes: number of classes
        :param optimizer: name of optimizer. See tf.keras.optimizers (default: 'adam')
        :param loss: name of objective function. See tf.keras.losses
        :param metrics: name of the metric to be evaluated by the model (default: 'accuracy')
        :param epochs: number of epochs to train the model
        """

        self.model = Sequential([
            data_augmentation_layer(size, size),
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

        :param model_name: name of the model. The path to the saved model will be './saved_model/model_name'
        """
        # model_name = self.data_type + self.task + model_name
        self.model.save('./saved_model/' + model_name)