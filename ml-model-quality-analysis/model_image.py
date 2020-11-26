import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from data import download_data
from robustness import get_gradient_sign
from model_interface import ModelInterface

class ModelImage(ModelInterface):
    """Implement the ModelInterface for an image classification model.

    Attributes:
    -----------
    data_type: data type, in this case 'image'
    task: machine learning task, in this case 'classification'
    model: machine learning model (loaded as a keras model)
    input_shape: input shape of the ML model
    num_classes: number of classes if the task is classification
    data: data to test the model with (loaded as a TF dataset)

    Methods:
    --------
    load_model(model_path=''): Load a pretrained model.
    load_data(data_url='', data_dir='', data_tf=''): Load and preprocess data.
    predict(adv=True): Generate the model predictions for the given data and for adversarial examples.
    """

    def __init__(self, data_type='image', task='classification'):
        super().__init__()
        self.data_type = data_type
        self.task = task
        self.model = None
        self.input_shape = None
        self.num_classes = None
        self.data = None


    def load_model(self, model_path: str):
        """Load a pretrained model. Save its input shape and output shape (number of classes in a classification task).

        :param model_path: path to the saved model folder
        """
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.get_layer(self.model.layers[0].name).input.shape
        self.num_classes = self.model.get_layer(self.model.layers[-1].name).output.shape[1]
        if self.num_classes == 1: self.num_classes = 2
        # print('Model loaded! Input shape:', self.input_shape, ', number of classes:', self.num_classes)

        return self


    def load_data(self, data_url='', data_dir='', data_tf='',
                  split=0.1, test_samples=100,
                  batch_size=1, shuffle=True):
        """Load and preprocess data.

        :param data_url: url download the data from
        :param data_dir: path to the directory containing the data
            This main directory should have subdirectories with the names of the classes
        :param data_tf: name of the TensorFlow dataset. See tfds.list_builders()
        :param split: percentage of samples for testing (default: 0.1)
        :param test_samples: number of samples to test the model (default: 100)
        :param batch_size: size of the batches of data (default: 32)
        :param shuffle: whether to shuffle the data (default: True)
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
        #print('Data loaded! ', data)

        return self


    def predict(self, num_examples=100, adv=False, epsilon=0.1, plot=False):
        """Generate the model predictions for the given data. Can create adversarial images and predict their labels.

        :param num_examples: number of examples to predict (default: 100)
        :param adv: whether to create adversarial examples and predict its labels (default: False)
        :param epsilon: multiplier to ensure the perturbations to the original image are small (default: 0.1)
        :param plot: whether to plot the original and adversarial images (default: False)

        :returns: y_true: labels to perform functionality and robustness analysis
        :returns: y_pred: predictions of the model for the original examples
        :returns: y_adv: predictions of the model for the adversarial examples
        """
        y_true = []
        y_pred = []
        y_adv = []

        for image, label in self.data.take(num_examples):

            # Store expected labels in y_true
            label_expected = label.numpy()[0]
            y_true.append(label_expected)

            # Make predictions and store the predicted label in y_pred
            model_pred = self.model.predict(image)
            if self.num_classes > 2:
                label_predicted = tf.argmax(model_pred[0]).numpy()
            else:
                label_predicted = tf.round(model_pred[0]).numpy()
            y_pred.append(label_predicted)


            # Create an adversarial example using the Fast Gradient Sign Method
            if adv:
                # Prepare label to calculate loss using binary or categorical cross-entropy
                if self.num_classes > 2:
                    label = tf.one_hot(label, model_pred.shape[-1])
                label = tf.reshape(label, (1, model_pred.shape[-1]))

                # Create perturbations
                perturbations = get_gradient_sign(self.model, self.num_classes, image, label)

                # Distort the original image
                image = image / 255
                adv_image = image + epsilon * perturbations
                adv_image = adv_image * 255

                # Make predictions for the adversarial image and store the predicted label in y_adv
                adv_pred = self.model.predict(adv_image)
                if self.num_classes > 2:
                    adv_label = tf.argmax(adv_pred[0]).numpy()
                else:
                    adv_label = tf.round(adv_pred[0]).numpy()
                y_adv.append(adv_label)


                # Plot the original image, the perturbations and the adversarial example
                if plot:
                    plt.figure(figsize=(11, 4))

                    plt.subplot(1, 3, 1)
                    plt.imshow(image[0])
                    plt.title('Original Image \nExpected label: %i / Predicted label: %i' % (label_expected, label_predicted))

                    plt.subplot(1, 3, 2)
                    perturbations = tf.clip_by_value(perturbations, 0, 1)
                    plt.imshow(perturbations[0])
                    plt.title('Perturbation')
                    # if channels == 1:
                    # plt.imshow(tf.reshape(perturbations[0], size))

                    plt.subplot(1, 3, 3)
                    adv_image = adv_image / 255
                    adv_image = tf.clip_by_value(adv_image, 0, 1)
                    plt.imshow(adv_image[0])
                    plt.title('Adversarial Example \nPredicted label: %i' % adv_label)
                    # if channels == 1:
                    # adv_x = tf.reshape(adv_x, size)

                    plt.suptitle("Adversarial Example")
                    plt.show()

        return y_true, y_pred, y_adv