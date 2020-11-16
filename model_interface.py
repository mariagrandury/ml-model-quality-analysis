"""Create an interface to an abstract ML model that is equipped with all the
methods that are relevant to enable the model to be used inside a quality analysis."""

class ModelInterface:

    def __init__(self):
        self.type = None
        self.task = None
        self.model = None
        self.data = None


    def load_model(self, model_path):
        """Load a pretrained model.

        :param model_path: Path to model file.
        """
        raise NotImplementedError

    def load_data(self, samples, size, data_path, data_url, data_tf):
        """Load and preprocess data.

        :param samples: Number of samples to test the model (default: 100)
        :param size: Size of the resized images (default: (150,150))
        :param data_path: Path to the directory containing the data (e.g. './data/predict/')
        :param data_url: URL to download the data (e.g. 'https:// example.zip')
        :param data_tf: Name of the TensorFlow dataset (default: 'cats_vs_dogs')
        """
        raise NotImplementedError


    def predict(self):
        """Generate the model predictions for the given data.

        :returns: prediction: Prediction of the model. Numpy array of shape (1,).
        :returns: labels: Labels to check if the predictions are correct.
        """
        raise NotImplementedError
