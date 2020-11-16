import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing import image

class ModelImage:

    def __init__(self):
        self.type = 'image'
        self.task = None
        self.model = None
        self.data = None


    def load_model(self, model_path='./saved_model/image-binary-classification-imagenet-acc979'):
        """Load a pretrained model.

        :param model_path: Path to model file.
        """
        self.model = tf.keras.models.load_model(model_path)


    def load_data(self, samples=100, size=(150,150), data_path=None, data_url=None, data_tf='cats_vs_dogs'):
        """Load and preprocess data.

        :param samples: Number of samples to test the model (default: 100)
        :param size: Size of the resized images (default: (150,150))
        :param data_path: Path to the directory containing the data (e.g. './data/predict/')
        :param data_url: URL to download the data (e.g. 'https:// example.zip')
        :param data_tf: Name of the TensorFlow dataset (default: 'cats_vs_dogs')
        """
        if data_path:
            for file in os.listdir(data_path):
                path = data_path + file
                img = image.load_img(path, target_size=size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                processed_data = np.vstack([x])

        if data_tf:
            split = "train[:100]"
            test_ds = tfds.load(data_tf, split=split, as_supervised=True, shuffle_files=True)
            input_data = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))
            processed_data = input_data.cache().batch(32).prefetch(buffer_size=10)

        self.data = processed_data


    def predict(self):
        """Generate the model predictions for the given data.

        :returns: prediction: Prediction of the model. Numpy array of shape (1,).
        :returns: labels: Labels to check if the predictions are correct.
        """
        predictions= []
        labels = []
        for image, label in tfds.as_numpy(self.data):
            predictions.append(self.model.predict(image))
            labels.append(label)

        y_pred = []
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
