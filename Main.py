import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

np.random.seed(10)

x_train = (x_train.astype(np.float32) - 127.5) / 127.5

x_train = x_train.reshape(-1, 784)
"""
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
"""

def create_generator():
    generator = tf.keras.models.Sequential()
    generator.add(tf.keras.layers.Dense(128, input_dim=100, activation=tf.nn.sigmoid))
    generator.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
    generator.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
    generator.add(tf.keras.layers.Dense(784, activation=tf.nn.sigmoid))

    generator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return generator

def create_discriminator():
    discriminator = tf.keras.models.Sequential()
    discriminator.add(tf.keras.layers.Flatten())
    discriminator.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
    discriminator.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
    discriminator.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return discriminator

discriminator = create_discriminator()
generator = create_generator()

discriminator.trainable = False

gan_input = tf.keras.layers.Input(shape=(100,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(optimizer="adam", loss="binary_crossentropy")

def show_images(noise):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))

    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        plt.imshow(image.reshape((28, 28)), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


for epoch in range(10):
    for batch in range(1000):
        noise = np.random.normal(0, 1, size=(16, 100))
        fake_x = generator.predict(noise)

        real_x = x_train[np.random.randint(0, x_train.shape[0], size=16)]

        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2*16)
        disc_y[:16] = 0.9

        d_loss = discriminator.train_on_batch(x, disc_y)

        y_gen = np.ones(16)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
    noise = np.random.normal(0, 1, size=(100, 100))
    show_images(noise)
