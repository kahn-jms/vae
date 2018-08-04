#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Keras callback plot VAE latent space at different epochs
# James Kahn 2018

import os
from keras.callbacks import Callback
import matplotlib.pyplot as plt


class plotLatentSpace(Callback):
    def __init__(self, output_dir, validation_x, validation_y, period=1):
        '''Want to add some extra input args to base CallBack class

        For now need to pass validation labels manually, cna't include it in
        the fit() validation call unless I figure out a way to tell the fit
        validation to ignore it
        '''
        super(plotLatentSpace, self).__init__()
        self.output_dir = output_dir
        self.period = period
        # This doesn't seem to be saved as a model attribute anymore
        # self.x_test = self.model.validation_data[0]
        self.x_test = validation_x
        self.y_test = validation_y

        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_begin(self, logs={}):
        self.encoder = self.model.get_layer('encoder')
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0:
            filename = os.path.join(self.output_dir, 'vae_latent_space_{}.pdf'.format(epoch))

            z_mean, _, _ = self.encoder.predict(self.x_test, batch_size=self.params['batch_size'])

            plt.figure()
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=self.y_test)
            plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.savefig(filename)
            # plt.show()

        return
