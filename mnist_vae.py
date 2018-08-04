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

from plotLatentSpace import plotLatentSpace


class vaeMNISTConv2D():
    def __init__(self):
        # Load MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.latent_dim = 2
        # Should be able to get this from x_train.shape
        self.input_dim = 28 * 28

        # self.encoder, self.encoder_inputs, self.z_mean, self.z_log_var = self.build_encoder(self.x_train.shape[1:])
        # self.decoder, self.decoder_outputs = self.build_decoder()
        self.vae_model = self.build_vae()
        #     self.encoder,
        #     self.decoder,
        #     self.encoder_inputs,
        #     self.decoder_outputs
        # )

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

    def build_vae(self):
    # def build_encoder(self, input_shape):

        # Start with normal convolutional net to classify images (kinda)
        encoder_input = Input(shape=self.x_train.shape[1:], name='encoder_input')
        encoder_1 = Conv2D(
            8,
            (3, 3),
            padding='same',
            # input_shape=input_shape
        )(encoder_input)
        encoder_2 = Conv2D(8, (3, 3), activation='relu')(encoder_1)
        encoder_3 = MaxPooling2D(pool_size=(2, 2))(encoder_2)
        encoder_4 = Dropout(0.25)(encoder_3)
        encoder_5 = Flatten()(encoder_4)
        encoder_6 = Dense(32, activation='relu')(encoder_5)
        encoder_7 = Dropout(0.25)(encoder_6)

        # Instead of classes, output the encoding as a mean and stddev
        z_mean = Dense(self.latent_dim, name='z_mean')(encoder_7)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(encoder_7)

        # Add our random sampler as a final layer of encoder
        # Would like to separate this to a separate sub-network
        z = Lambda(
            self.latent_sampling,
            output_shape=(self.latent_dim,),
            name='z'
        )([z_mean, z_log_var])

        # Finally compile the model
        encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # return encoder, c_input, z_mean, z_log_var

    # def build_decoder(self):
        '''Basically the encoder backwards

        But this time need to calculate  upsampling sizes to get the final pixel sizes right.
        '''

        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')
        decoder_1 = Dense(7 * 7 * 16, activation='relu')(decoder_input)
        decoder_2 = Reshape((7, 7, 16))(decoder_1)
        # Upsample to (14, 14, ...)
        decoder_3 = Conv2DTranspose(
            16,
            5,
            strides=2,
            padding='same',
            activation='relu',
            kernel_initializer='glorot_normal'
        )(decoder_2)
        decoder_4 = BatchNormalization()(decoder_3)
        # Upsample to (28, 28, ...)
        decoder_output = Conv2DTranspose(
            1,
            5,
            strides=2,
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_normal'
        )(decoder_4)

        # Finally compile the model
        decoder = Model(decoder_input, decoder_output, name='decoder')
        decoder.summary()

        # return decoder, c4

        '''Instantiate the VAE model'''
        # enc = encoder(encoder_inputs)[2]
        # enc = K.reshape(enc, (64, 2))
        # print('here:', enc)
        # outputs = decoder(enc)
        outputs = decoder(encoder(encoder_input)[2])
        vae = Model(encoder_input, outputs, name='vae_mlp')
        vae_loss = self.build_loss(encoder_input, outputs, z_mean, z_log_var)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()

        return vae

    def build_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = mse(inputs, outputs)
        # print(reconstruction_loss)
        reconstruction_loss *= self.input_dim
        # Might need to add axis=[1, 2]
        reconstruction_loss = K.sum(reconstruction_loss)
        kl_loss = (1 + z_log_var -
                   K.square(z_mean) -
                   K.exp(z_log_var))
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        print('recloss:', reconstruction_loss)
        print('klloss:', kl_loss)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss


if __name__ == '__main__':

    epochs = 1
    batch_size = 128
    # Create our training model
    vae = vaeMNISTConv2D()

    # Set up necessary callback
    plotLatentCallback = plotLatentSpace('latent_plots', vae.y_test)

    vae.vae_model.fit(
        vae.x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(vae.x_test, None),
        callbacks=[plotLatentCallback],
    )
    vae.vae_model.save_weights('trainings/vae_mlp_mnist.h5')
