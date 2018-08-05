#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Keras callback plot VAE latent space at different epochs
# James Kahn 2018

import os
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np


class plotDecoderSamples(Callback):
    def __init__(self, output_dir, n_samples=10, period=1):
        '''Want to add some extra input args to base CallBack class

        For now need to pass validation data manually, cna't include it in
        the fit() validation call unless I figure out a way to tell the fit
        validation to ignore it
        '''
        super(plotDecoderSamples, self).__init__()
        self.output_dir = output_dir
        self.n_samples = n_samples
        self.period = period
        # This doesn't seem to be saved as a model attribute anymore
        # self.x_test = self.model.validation_data[0]
        # self.x_test = validation_x
        # self.y_test = validation_y

        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_begin(self, logs={}):
        self.decoder = self.model.get_layer('decoder')
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0:
            filename = os.path.join(self.output_dir, 'vae_decoder_sample_{}.pdf'.format(epoch))

            # Cut the latent space into a grid and we will draw samples from each point
            grid_x = np.linspace(-4, 4, self.n_samples)
            grid_y = np.linspace(-4, 4, self.n_samples)  # [::-1]

            fig, axarr = plt.subplots(self.n_samples, self.n_samples)

            for idx, x in enumerate(grid_x):
                for idy, y in enumerate(grid_y):
                    z_sample = np.array([[x, y]])
                    sample_fig = self.decoder.predict(z_sample)[0]
                    sample_fig = sample_fig.reshape(28, 28)
                    # I think matplotlib scales ranges [0, 1] itself
                    # sample_fig *= 255
                    axarr[idx, idy].imshow(sample_fig, cmap='Greys_r')
                    # axarr[idx, idy].plot(sample_fig)
                    # Could also do ax.plot(sample_fig) ?
                    # axarr[idx, idy].set_title('[{}, {}]'.format(x, y))

            # Set up the x, y labels to show the z coords we sampled
            # Need to figure out how to do this
            # plt.xlabel("z[0]")
            # plt.ylabel("z[1]")
            plt.savefig(filename)
            # plt.show()

        return
