"""Helper functions to perform robustness analysis."""


import tensorflow as tf


def get_gradient_sign(model, num_classes, input_image, input_label):
    """Get the sign of the gradient of the loss with respect to the input image. Helper function to create an
        adversarial example using the Fast Gradient Sign Method.

    :param model: model used to make predictions
    :param num_classes: number of classes
    :param input_image: original image that will be distorted to create the adversarial image
    :param input_label: label of the original image, must be one hot

    :returns signed_grad: sign of the gradient of the loss with respect to the input image
    """

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        if num_classes > 2:
            loss = tf.keras.losses.categorical_crossentropy(input_label, prediction, from_logits=True)
        else:
            loss = tf.keras.losses.binary_crossentropy(input_label, prediction, from_logits=True)

    gradient = tape.gradient(loss, input_image)

    signed_grad = tf.sign(gradient)

    return signed_grad