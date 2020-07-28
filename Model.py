import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Reshape, UpSampling2D,
                          ZeroPadding2D)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from PIL import Image, ImageDraw

# Consistent results
np.random.seed(10)

# The dimension of z
noise_dim = 100

epochs = 1000

save_path = 'outputs'

img_rows, img_cols, channels = 28, 28, 1

optimizer = Adam(0.0001, beta_1=0.5)

# Create path for saving images
def save_original():
    counter = 0
    plt.figure(figsize=(5, 5))
    for filename in os.listdir("handwriting_samples"):
        if (filename.endswith(".png") and counter < 25):
            original_image = cv2.imread(f'handwriting_samples/{filename}')
            plt.subplot(5,5,counter+1)
            plt.imshow(original_image)
            plt.axis('off')
            counter += 1

    plt.tight_layout()
    plt.savefig(f'{save_path}/original.png')

if not os.path.isdir(save_path):
    os.mkdir(save_path)

save_original()

reg_array = []
for filename in os.listdir("handwriting_samples"):
    if (filename.endswith(".png")) :
        image = cv2.imread(f'handwriting_samples/{filename}')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reg_array.append(gray_image)
x_train = np.array(reg_array)


batch_size = 8
steps_per_epoch = int(len(reg_array)/batch_size)

x_train = (x_train.astype(np.float32) - 127.5) / 127.5

x_train = x_train.reshape(-1, img_rows*img_cols*channels)
print(x_train.ndim, x_train.shape, x_train.size)


def create_generator():
    generator = Sequential()
    generator.add(Dense(256, input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())

    generator.add(Dense(img_rows*img_cols*channels, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def create_descriminator():
    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=img_rows*img_cols*channels))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))


    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))


    discriminator.add(Dense(1, activation="sigmoid"))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

discriminator = create_descriminator()
generator = create_generator()

discriminator.trainable = False

gan_input = Input(shape=(noise_dim,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

gan.summary()

def show_images(noise, epoch=None):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(5, 5))

    for i, image in enumerate(generated_images):
        plt.subplot(5, 5, i+1)
        if channels == 1:
            plt.imshow(image.reshape((img_rows, img_cols)), cmap='gray')
        else:
            plt.imshow(image.reshape((img_rows, img_cols, channels)))
        plt.axis('off')

    plt.tight_layout()

    if epoch != None:
        plt.savefig(f'{save_path}/gan-images_epoch-{epoch}.png')

# Constant noise for viewing how the GAN progresses
static_noise = np.random.normal(0, 1, size=(25, noise_dim))

for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_x = generator.predict(noise)

        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        """
        disc_y_real = np.ones(batch_size)
        disc_y_real[0:] = 0.9
        d_loss_real = discriminator.train_on_batch(real_x, disc_y_real)
        disc_y_fake = np.zeros(batch_size)
        d_loss_fake = discriminator.train_on_batch(fake_x, disc_y_fake)
        d_loss = 0
        """


        x = np.concatenate((real_x, fake_x))
        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9
        d_loss = discriminator.train_on_batch(x, disc_y)


        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

    if epoch%5==0:
        print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
        show_images(static_noise, epoch)
