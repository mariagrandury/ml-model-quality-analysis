import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds

from data import download_data
from robustness import get_gradient_sign
import matplotlib.pyplot as plt

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
        if self.num_classes == 1: self.num_classes = 2
        print('Model loaded! Input shape:', self.input_shape, ', number of classes:', self.num_classes)

        return self


    def load_data(self, split=0.1, test_samples=100,
                  batch_size=1, shuffle=True,
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
        seed = 23 # for reproducibility
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


    def predict(self, num_examples=100, adv=True, show=False):
        """Generate the model predictions for the given data.

        :param num_examples: Int, number of examples to predict (default: 100)
        :param adv: Bool, whether to create adversarial examples (default: True)
        :param show: Bool, whether to show an adversarial example (default: False)

        :returns: y_true: Labels to perform functionality and robustness analysis.
        :returns: y_pred: Predictions of the model for the original examples.
        :returns: y_adv: Predictions of the model for the adversarial examples.
        """
        y_true = []
        y_pred = []
        y_adv = []

        for image, label in self.data.take(num_examples):
            label_original = label.numpy()[0]
            y_true.append(label_original)

            model_pred = self.model.predict(image)
            if self.num_classes > 2:
                label_predicted = tf.argmax(model_pred[0]).numpy()
            else:
                label_predicted = tf.round(model_pred[0]).numpy()
            y_pred.append(label_predicted)

            if adv:
                if self.num_classes > 2:
                    label = tf.one_hot(label, model_pred.shape[-1])
                    label = tf.reshape(label, (1, model_pred.shape[-1]))
                else:
                    label = label

                perturbations = []
                perturbations = get_gradient_sign(self.model, image, label)

                image = image / 255
                adv_x = image + 0.15 * perturbations
                adv_x = adv_x * 255

                adv_pred = self.model.predict(adv_x)

                if self.num_classes > 2:
                    adv_label = tf.argmax(adv_pred[0]).numpy()
                else:
                    adv_label = tf.round(adv_pred[0]).numpy()
                y_adv.append(adv_label)

            if show:
                plt.figure(figsize=(11, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(image[0])
                plt.title('Original Image, label: %i / %i' % (label_original, label_predicted))

                plt.subplot(1, 3, 2)
                perturbations = tf.clip_by_value(perturbations, 0, 1)
                plt.imshow(perturbations[0])
                plt.title('Perturbation')
                # if channels == 1:
                # plt.imshow(tf.reshape(perturbations[0], size))

                plt.subplot(1, 3, 3)
                adv_x = adv_x / 255
                adv_x = tf.clip_by_value(adv_x, 0, 1)
                plt.imshow(adv_x[0])
                plt.title('Adversarial Example, label: %i' % adv_label)
                # if channels == 1:
                # adv_x = tf.reshape(adv_x, size)

                plt.suptitle("Adversarial Example")
                plt.show()

        return y_true, y_pred, y_adv