import tensorflow as tf


def get_gradient_sign(model, input_image, input_label):
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)

    signed_grad = tf.sign(gradient)

    return signed_grad