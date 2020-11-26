class ModelInterface:
    """Interface to an abstract ML model that is equipped with relevant methods to enable the model
    to be used inside a quality analysis.

    Attributes:
    -----------
    data_type: data type, e.g. 'image' or 'text'
    task: machine learning task, e.g. 'classification' or 'regression'
    model: machine learning model
    data: data to test the model with

    Methods:
    --------
    load_model(): load a pretrained model
    load_data(): load and preprocess data
    predict(): generate the model predictions
    """

    def __init__(self):
        self.type = None
        self.task = None
        self.model = None
        self.data = None


    def load_model(self, model_path):
        """Load a pretrained model.

        :param model_path: path to the saved model folder
        """
        raise NotImplementedError


    def load_data(self, data_source, test_samples):
        """Load and preprocess data.

        :param data_source: source of the testing data
        :param test_samples: number of samples to test the model
        """
        raise NotImplementedError


    def predict(self):
        """Generate the model predictions for the given data.

        :returns: prediction: predictions of the model
        :returns: labels: labels to check if the predictions are correct
        """
        raise NotImplementedError
