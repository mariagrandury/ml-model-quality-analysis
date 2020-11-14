"""Create an interface to an abstract ML model that is equipped with all the
methods that are relevant to enable the model to be used inside a quality analysis."""

class ModelInterface:

    def __init__(self):
        self.model = None

    def load_data(self, data_path, data_url):
        """Load data

        :param data_path: Path to the directory containing the data if the data is already downloaded.
        :param data_url: URL to download the data.
        """
        raise NotImplementedError


    def preprocess_data(self, input_data):
        """Preprocess raw input data.


        :param input_data: Batch of samples. Numpy array of shape (m, n) where m is
                the number of samples and n is the dimension of the feature vector.
        """
        raise NotImplementedError


    def split_data(self, percentages):
        """Split data into train, dev and test set.


        :param percentages: Array of floats in range [0,1] that determine the splitting.
        """
        raise NotImplementedError


    def load_model(self, model_path):
        """Load a pretrained model.


        :param model_path: Path to model file.
        """
        raise NotImplementedError


    def fit(self, x_train, y_train, optimizer, learning_rate, loss, epochs, steps_per_epoch):
        """Fit a model on a training dataset.


        :param x_train: Batch of samples for training. Numpy array of shape (m, n),
            where m is the number of samples and n is the dimension of the feature vector.
        :param y_train: Observed training targets. Numpy array of shape (n,).
        :param optimizer: Optimizer
        :param learning_rate: Learning rate
        :param loss: Loss function
        :param epochs: Number of epochs
        :param steps_per_epoch: Number of steps per epoch
        """
        raise NotImplementedError


    def save(self):
        """Save the model"""
        raise NotImplementedError


    def predict(self, input_features):
        """Function that accepts a model and input data and returns a prediction.


        :param input_features: Features required by the model to generate prediction.
            Numpy array of shape (1, n) where n is the dimension of the feature vector.

        :returns: prediction: Prediction of the model. Numpy array of shape (1,).
        """
        raise NotImplementedError
