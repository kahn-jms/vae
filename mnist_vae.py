#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Variational autoencoder to run on MNIST images
# James Kahn 2018

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Lambda, Input, Reshape
from keras.losses import mse
from keras import backend as K


class vaeMNISTConv2D():
    def __init__(self):
        # Load MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.latent_dim = 2
        # Should be able to get this from x_train.shape
        self.input_dim = 28 * 28

        self.encoder, self.encoder_inputs, self.z_mean, self.z_log_var = self.build_encoder(self.x_train.shape[1:])
        self.decoder, self.decoder_outputs = self.build_decoder()
        self.vae_model = self.build_vae(
            self.encoder,
            self.decoder,
            self.encoder_inputs,
            self.decoder_outputs
        )

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train = K.expand_dims(x_train, -1)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        print('Train shape:', x_train.shape)

        # Normalise data
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        return (x_train, y_train), (x_test, y_test)

    def latent_sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        # Since it's a log the 0.5 is a square root
        return z_mean + (K.exp(0.5 * z_log_var) * epsilon)

    def build_encoder(self, input_shape):

        # Start with normal convolutional net to classify images (kinda)
        c_input = Input(shape=input_shape, name='encoder_input')
        c1 = Conv2D(
            8,
            (3, 3),
            padding='same',
            input_shape=input_shape
        )(c_input)
        c2 = Conv2D(8, (3, 3), activation='relu')(c1)
        c3 = MaxPooling2D(pool_size=(2, 2))(c2)
        c4 = Dropout(0.25)(c3)
        c5 = Flatten()(c4)
        c6 = Dense(32, activation='relu')(c5)
        c7 = Dropout(0.25)(c6)

        # Instead of classes, output the encoding as a mean and stddev
        z_mean = Dense(self.latent_dim, name='z_mean')(c7)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(c7)

        # Add our random sampler as a final layer of encoder
        # Would like to separate this to a separate sub-network
        z = Lambda(
            self.latent_sampling,
            output_shape=(self.latent_dim,),
            name='z'
        )([z_mean, z_log_var])

        # Finally compile the model
        encoder = Model(c_input, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        return encoder, c_input, z_mean, z_log_var

    def build_decoder(self):
        '''Basically the encoder backwards

        But this time need to calculate  upsampling sizes to get the final pixel sizes right.
        '''

        c_input = Input(shape=(self.latent_dim,), name='decoder_input')
        c1 = Dense(7 * 7 * 16, activation='relu')(c_input)
        c2 = Reshape((7, 7, 16))(c1)
        # Upsample to (14, 14, ...)
        c3 = Conv2DTranspose(
            16,
            5,
            strides=2,
            padding='same',
            activation='relu',
            kernel_initializer='glorot_normal'
        )(c2)
        c3 = BatchNormalization()(c3)
        # Upsample to (28, 28, ...)
        c4 = Conv2DTranspose(
            1,
            5,
            strides=2,
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_normal'
        )(c3)

        # Finally compile the model
        decoder = Model(c_input, c4, name='decoder')
        decoder.summary()

        return decoder, c4

    def build_vae(self, encoder, decoder, encoder_inputs, decoder_outputs):
        '''Instantiate the VAE model'''
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = Model(encoder_inputs, outputs, name='vae_mlp')
        vae_loss = self.build_loss()
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()

        return vae

    def build_loss(self):
        reconstruction_loss = mse(self.encoder_inputs, self.decoder_outputs)
        reconstruction_loss *= self.input_dim
        kl_loss = (1 + self.z_log_var -
                   K.square(self.z_mean) -
                   K.exp(self.z_log_var))
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss


if __name__ == '__main__':

    epochs = 10
    batch_size = 64
    # Create our training model
    vae = vaeMNISTConv2D()

    vae.vae_model.fit(
        vae.x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(vae.x_test, None),
    )
