import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds

from data import download_data


class ModelImage:

    def __init__(self):
        self.type = 'image'
        self.task = None
        self.model = None
        self.input_shape = None
        self.num_classes = None
        self.data = None


    def load_model(self, model_path):
        """Load a pretrained model.

        :param model_path: Str, path to saved model folder.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.get_layer(self.model.layers[0].name).input.shape
        self.num_classes = self.model.get_layer(self.model.layers[-1].name).output.shape[1]
        print('Model loaded! Input shape:', self.input_shape, ', number of classes:', self.num_classes)


    def load_data(self, split=0.1, test_samples=100,
                  batch_size=32, shuffle=True,
                  data_url=None, data_dir=None, data_tf=None):
        """Load and preprocess data.

        :param split: Float, percentage of samples for testing (default: 0.1)
        :param test_samples: Int, number of samples to test the model (default: 100)
        :param batch_size: Int, size of the batches of data (default: 32)
        :param shuffle: Bool, whether to shuffle the data (default: True)
        :param data_url: Str, url to the zip or tar file to download the data.
        :param data_dir: Str, path to the directory containing the data.
            This main directory should have subdirectories with the names of the classes.
        :param data_tf: Str, name of the TensorFlow dataset. See tfds.list_builders().
        """
        seed = 123 # for reproducibility
        AUTOTUNE = tf.data.experimental.AUTOTUNE # for better performance
        size = (self.input_shape[1], self.input_shape[2]) # size to resize images


        # Download data from url
        if data_url:
            data_dir = download_data(data_url, cache_dir='./')
            print('Data downloaded!')


        # Load data from directory
        if data_dir:
            data_dir = pathlib.Path(data_dir)
            total = len(list(data_dir.glob('*/*.jpg')))
            if test_samples: split = test_samples / total

            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir, validation_split=split, subset='validation', seed=seed,
                image_size=size, batch_size=batch_size, shuffle=shuffle
            )
            data = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


        # Load tensorflow dataset
        if data_tf:
            split = "train[:" + str(test_samples) + "]"
            test_ds = tfds.load(data_tf, split=split, as_supervised=True, shuffle_files=shuffle)
            test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))
            data = test_ds.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)


        self.data = data
        print('Data loaded! ', data)


    def predict(self):
        """Generate the model predictions for the given data.

        :returns: predictions: Predictions of the model. Numpy array of shape (1,).
        :returns: labels: Labels to perform functionality analysis.
        """
        predictions= []
        labels = []
        for image, label in tfds.as_numpy(self.data):
            predictions.append(self.model.predict(image))
            labels.append(label)

        y_pred = []
        if self.num_classes > 2:
            for batch in predictions:
                for prediction in batch:
                    y_pred.append(tf.argmax(prediction).numpy())
        else:
            for batch in predictions:
                for prediction in batch:
                    if prediction < 0.5:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)

        y_true = []
        for batch in labels:
            for label in batch:
                y_true.append(label)

        return y_pred, y_true
