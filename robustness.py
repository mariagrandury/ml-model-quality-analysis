import tensorflow as tf


def get_gradient_sign(model, input_image, input_label):

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.categorical_crossentropy(input_label, prediction, from_logits=True)

    gradient = tape.gradient(loss, input_image)

    signed_grad = tf.sign(gradient)

    return signed_grad