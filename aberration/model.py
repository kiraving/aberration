"""
Modified from https://github.com/mpicbg-csbd/phasenet/
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
See full license at the end of the document
"""

import numpy as np
from distutils.version import LooseVersion
import warnings

import keras
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from csbdeep.utils import _raise, axes_check_and_normalize, normalize
from csbdeep.models import BaseConfig, BaseModel

from phasenet.psf import PsfGenerator3D
from phasenet.zernike import random_zernike_wavefront, ensure_dict
from phasenet.noise import add_random_noise
from phasenet.utils import cropper3D

from .phantoms import Phantom3D

from scipy.signal import convolve
from scipy.ndimage.filters import gaussian_filter

import secrets

from pathlib import Path




class Data:

    def __init__(self,
                 amplitude_ranges, order='ansi', normed=True,
                 batch_size=1,
                 psf_shape=(64, 64, 64), units=(0.1, 0.1, 0.1), na_detection=1.1, lam_detection=.5, n=1.33, n_threads=4,
                 noise_snr=None, noise_mean=None, noise_sigma=None, noise_perlin_flag=False, gaussian_blur_sigma=None,
                 phantom_params=None,
                 initcrop=None,
                 crop_shape=None, jitter=False, max_jitter=None,
                 planes=None
                 ):
        """Encapsulates data generation
        :param order: string, Zernike nomenclature used when the amplitude ranges are given as a list.
            Highly recommended to use Ansi. Usage of Noll requires extra manual corrections
        :param psf_shape: tuple, shape of psf, eg (32,32,32)
        :param units: tuple, units in microns, eg (0.1,0.1,0.1)
        :param lam_detection: scalar, wavelength in micrometer, eg 0.5
        :param n: scalar, refractive index, eg 1.33
        :param na_detection: scalar, numerical aperture of detection objective, eg 1.1
        :param n_threads: integer, for multiprocessing
        :param noise_snr: scalar or tuple, signal to noise ratio
        :param noise_mean: scalar or tuple, mean background noise
        :param noise_sigma: scalar or tuple, sigma for Gaussian noise
        :param noise_perlin_flag: boolean, default is False
        :param gaussian_blur_sigma: float, sigma for Gaussian blurring after adding noise, default is None
        :param phantom_params : dictionary, phantom name and parameters for that phantom
        :param crop_shape: tuple, crop shape
        :param initcrop: tuple, the input shape after preprocessing (e.g. manual cropping of originally large images). Set it if you want to select z-planes but avoid cropping on x and y
        Recommendation 1: Set crop_shape manually. If both crop_shape and inicrop are None, crop_shape = psf_shape (the image is cropped to the size of the psf),
                        2: Pass input images larger than the psf and the crop shape so that the boundary effects are cut by cropping
        :param jitter: Boolean, randomly move the center point within a given limit, default is False
        :param max_jitter: tuple, maximum displacement for jitter, if None then it gets a default value
        :param planes: list, z planes with respect to center, if None then it takes all the planes
        """

        self.psfgen = PsfGenerator3D(psf_shape=psf_shape, units=units, lam_detection=lam_detection, n=n,
                                     na_detection=na_detection, n_threads=n_threads)
        self.order = order
        self.normed = normed
        self.amplitude_ranges = ensure_dict(amplitude_ranges, order)
        self.batch_size = batch_size
        self.snr = noise_snr
        self.sigma = noise_sigma
        self.mean = noise_mean
        self.intcrop = initcrop
        # initcrop = (100,100,100)
        self.noise_perlin_flag = noise_perlin_flag
        self.gaussian_blur_sigma = gaussian_blur_sigma
        if planes is not None:
            self.planes = np.array(planes)
            if crop_shape is None:
                if initcrop is not None:
                    self.crop_shape = tuple((self.planes.shape[0], initcrop[1], initcrop[2]))
                else:
                    self.crop_shape = tuple((self.planes.shape[0], psf_shape[1], psf_shape[2]))
            else:
                self.crop_shape = tuple((self.planes.shape[0], crop_shape[1], crop_shape[2]))
        else:
            self.planes = planes
            self.crop_shape = crop_shape
        self.jitter = jitter
        self.max_jitter = max_jitter
        self.phantom_params = phantom_params
        self.phantom = None
        self.psf_shape = psf_shape
        self.units = units

    def _single_psf(self):
        phi = random_zernike_wavefront(self.amplitude_ranges, order=self.order)
        psf = self.psfgen.incoherent_psf(phi, normed=self.normed)

        if self.phantom_params is not None:
        # Code for multiple phantoms
            if isinstance(self.phantom_params, list):
                # randomly select one element from the list

                self.phantom = secrets.choice(self.phantom_params)
            else:
                self.phantom = self.phantom_params
            #print("phantom: ",self.phantom)
            self.phantom.setdefault('shape', self.psf_shape)
            self.phantom.setdefault('units', self.units)
            self.phantom = Phantom3D.instantiate(**self.phantom)

            self.phantom.generate()
            obj = self.phantom.get()

            psf = convolve(obj, psf, 'same')

        if self.snr is not None and self.sigma is not None and self.mean is not None:
            self.noise_flag = True
            psf = add_random_noise(psf, self.snr, self.mean, self.sigma, perlin=self.noise_perlin_flag)
            if self.gaussian_blur_sigma is not None:
                gaussian_blur = (self.gaussian_blur_sigma, self.gaussian_blur_sigma) if np.isscalar(
                    self.gaussian_blur_sigma) else self.gaussian_blur_sigma
                gaussian_blur = np.random.uniform(*gaussian_blur)
                psf = gaussian_filter(psf, gaussian_blur)
        else:
            self.noise_flag = False
            if self.snr is not None or self.sigma is not None or self.mean is not None:
                warnings.warn("No noise added")

        if self.crop_shape is not None:
            self.crop_flag = True
            psf = cropper3D(psf, self.crop_shape, jitter=self.jitter, max_jitter=self.max_jitter, planes=self.planes)
        else:
            self.crop_flag = False

        return psf, phi.amplitudes_requested

    def generator(self):
        while True:
            psfs, amplitudes = zip(*(self._single_psf() for _ in range(self.batch_size)))
            psfs = [normalize(psf) for psf in psfs]
            X = np.expand_dims(np.stack(psfs, axis=0), -1)
            Y = np.stack(amplitudes, axis=0)
            yield X, Y


class Config(BaseConfig):
    """Configuration for phasenet models

    :param zernike_amplitude_ranges: dictionary or list, the values should either a scalar indicating the absolute magnitude
            or a tuple with upper and lower bound.
    :param zernike_order: string, Zernike nomenclature used when the amplitude ranges are given as a list.
    :param zernike_normed: Boolean, whether the Zernike are normalized according, default is True
    :param net_architecture: convnet or resnet, default is convnet
    :param net_kernel_size: convolution kernel size, default is (3,3,3)
    :param net_pool_size: max pool kernel size, default is (1,2,2)
    :param net_activation: activation, default is 'tanh'
    :param net_padding: padding for convolution, default is 'same'
    :param psf_shape: tuple, shpae of the psf, default is (64,64,64)
    :param psf_units: tuple, voxel unit (z,y,x) in um, default is (0.1,0.1,0.1)
    :param psf_na_detection: scalar, numerical aperture default is 1.1
    :param psf_lam_detection: scalar, wavelength in um, default is 0.5
    :param psf_n: scalar, refractive index of immersion medium, default is 1.33
    :param noise_snr: scalar or tuple, signal to noise ratio
    :param noise_mean: scalar or tuple, mean background noise
    :param noise_sigma: scalar or tuple, sigma for Gaussian noise
    :param gaussian_blur_sigma: float, sigma for Gaussian blurring after adding noise, default is None
    :param phantom_params: dictionary, parameters for the chosen phantom, e.g. {'name':'sphere','radius':0.1}
    :param crop_shape: tuple, crop shape
    :param jitter: Boolean, randomly move the center point within a given limit, default is False
    :param max_jitter: tuple, maximum displacement for jitter, if None then it gets a default value
    :param train_loss: string, training loss, default is 'mse'
    :param train_epochs: integer, number of epochs for training, default is 400
    :param train_steps_per_epoch: integer, number of steps per epoch, default is 5
    :param train_learning_rate: scalar, leaning rate, default is 0.0003
    :param train_batch_size: integer, batch size for training the network, default is 8
    :param train_n_val: integer, number of validation data, default is 128
    :param train_tensorboard: boolean, create tensor-board, default is True
    :param planes: list, z planes with respect to center, if None then it takes all the planes

    """

    def __init__(self, axes='ZYX', n_channel_in=1,**kwargs):
        """See class docstring."""

        super().__init__(axes=axes, n_channel_in=n_channel_in,n_channel_out=1)

        # default config (can be overwritten by kwargs below)
        self.zernike_amplitude_ranges  = {'vertical coma': (-0.1,0.1)} # Only within 0.1 range the results are reliable
        self.zernike_order             = 'ansi' # Noll is not recommended!
        self.zernike_normed            = True

        self.net_architecture         = "convnet"
        self.net_kernel_size           = (3,3,3)
        self.net_pool_size             = (1,2,2)
        self.net_activation            = 'tanh'
        self.net_padding               = 'same'

        self.psf_shape                 = (64,64,64)
        self.psf_units                 = (0.1,0.1,0.1)
        self.psf_na_detection          = 1.1
        self.psf_lam_detection         = 0.5
        self.psf_n                     = 1.33
        self.noise_mean                = 100
        self.noise_sigma               = 3.5
        self.noise_snr                 = (1.,5)
        self.noise_perlin_flag         = False
        self.gaussian_blur_sigma       = None
        self.phantom_params            = {'name':'points', 'num':1}
        self.crop_shape                = (32,32,32)
        self.jitter                    = True
        self.max_jitter                = None
        self.planes                    = None

        self.train_loss                = 'mse'
        self.train_epochs              = 400
        self.train_steps_per_epoch     = 5
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 8
        self.train_n_val               = 128
        self.train_tensorboard         = True
        #self.initcrop = initcrop #(100,100,100)
        
        # remove derived attributes that shouldn't be overwritten
        for k in ('n_dim', 'n_channel_out'):
            try: del kwargs[k]
            except KeyError: pass

        self.update_parameters(False, **kwargs)

        self.n_channel_out = len(random_zernike_wavefront(self.zernike_amplitude_ranges))


class PhaseNet(BaseModel):
    """PhaseNet model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    @property
    def _config_class(self):
        return Config


    @property
    def _axes_out(self):
        return 'C'

    def get_model_input_shape(self):

        if self.config.planes is not None:
            _p = np.array(self.config.planes)
            if self.config.crop_shape is None:

                # added functionality with initcrop
                if self.config.initcrop is not None:
                    config.model_input_shape = tuple((_p.shape[0], initcrop[1], initcrop[2]))
                else:
                    config.model_input_shape = tuple((_p.shape[0],self.config.psf_shape[1],self.config.psf_shape[2]))

            else:
                model_input_shape = tuple((_p.shape[0],self.config.crop_shape[1],self.config.crop_shape[2]))
        elif self.config.crop_shape is not None:
            model_input_shape = self.config.crop_shape
        else :
            model_input_shape = initcrop
        return model_input_shape

    def _build(self):
        model_input_shape = self.get_model_input_shape()
        input_shape = tuple(model_input_shape) + (self.config.n_channel_in,)
        output_size = self.config.n_channel_out
        kernel_size = self.config.net_kernel_size
        pool_size = self.config.net_pool_size
        activation = self.config.net_activation
        padding = self.config.net_padding

        if self.config.net_architecture == 'resnet':
            return self._resnet(input_shape, output_size, kernel_size, activation, padding, pool_size)
        elif self.config.net_architecture == 'convnet':
            return self._convnet(input_shape, output_size, kernel_size, activation, padding, pool_size)
        else:
            _raise(ValueError("Network architecture not defined"))

    def _convnet(self, input_shape, output_size, kernel_size, activation, padding, pool_size):

        inp = Input(name='X', shape=input_shape)
        t = Conv3D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding)(inp)
        t = Conv3D(8, name='conv2', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool1', pool_size=pool_size)(t)
        t = Conv3D(16, name='conv3', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(16, name='conv4', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool2', pool_size=pool_size)(t)
        t = Conv3D(32, name='conv5', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(32, name='conv6', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool3', pool_size=pool_size)(t)
        t = Conv3D(64, name='conv7', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(64, name='conv8', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool4', pool_size=pool_size)(t)
        t = Conv3D(128, name='conv9', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(128, name='conv10', kernel_size=kernel_size, activation=activation, padding=padding)(t)

        if input_shape[0] == 1:
            t = MaxPooling3D(name='maxpool5', pool_size=(1, 2, 2))(t)
        else:
            t = MaxPooling3D(name='maxpool5', pool_size=(2, 2, 2))(t)
        t = Flatten(name='flat')(t)
        t = Dense(64, name='dense1', activation=activation)(t)
        t = Dense(64, name='dense2', activation=activation)(t)

        oup = Dense(output_size, name='Y', activation='linear')(t)

        return Model(inputs=inp, outputs=oup)


    def _resnet(self, input_shape, output_size, kernel_size, activation, padding, pool_size):

        def resnet_block(n_filters, kernel_size=kernel_size, batch_norm=True, downsample=False, kernel_initializer="he_normal"):
            def f(inp):
                strides = (2, 2, 2) if downsample else (1, 1, 1)
                x = Conv3D(n_filters, kernel_size, padding='same', use_bias=(not batch_norm),
                           kernel_initializer=kernel_initializer, strides=strides)(inp)
                if batch_norm:
                    x = BatchNormalization()(x)
                x = Activation(activation)(x)

                x = Conv3D(n_filters, kernel_size, padding=padding, use_bias=(not batch_norm),kernel_initializer=kernel_initializer)(x)
                if batch_norm:
                    x = BatchNormalization()(x)

                if downsample:
                    inp = Conv3D(n_filters, (1, 1, 1), padding=padding, use_bias=(not batch_norm), kernel_initializer=kernel_initializer,
                                 strides=strides)(inp)
                    if batch_norm:
                        inp = BatchNormalization()(inp)

                x = Add()([inp, x])
                x = Activation(activation)(x)
                return x

            return f

        inp = Input(input_shape, name='X')
        x = inp
        n = 16

        depth = 3
        for i in range(depth):
            x = resnet_block(n * (2 ** i), downsample=(i > 0))(x)
            x = resnet_block(n * (2 ** i))(x)
        x = GlobalAveragePooling3D()(x)
        oup = Dense(output_size, name='Y')(x)

        return Model(inp, oup)


    def prepare_for_training(self, optimizer=None):
        """Prepare for neural network training.

        Compiles the model and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.

        """
        if optimizer is None:
            optimizer = Adam(lr=self.config.train_learning_rate)

        self.keras_model.compile(optimizer, loss=self.config.train_loss)

        self.callbacks = []
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                self.callbacks.append(TensorBoard(log_dir=str(self.logdir), write_graph=False))

        self._model_prepared = True


    def train(self, seed=None, epochs=None, steps_per_epoch=None, data_val = None):
        """Train the neural network with the given data.

        Parameters
        ----------
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            amplitude_ranges     = self.config.zernike_amplitude_ranges,
            order                = self.config.zernike_order,
            normed               = self.config.zernike_normed,
            psf_shape            = self.config.psf_shape,
            units                = self.config.psf_units,
            na_detection         = self.config.psf_na_detection,
            lam_detection        = self.config.psf_lam_detection,
            n                    = self.config.psf_n,
            noise_snr            = self.config.noise_snr,
            noise_mean           = self.config.noise_mean,
            noise_sigma          = self.config.noise_sigma,
            noise_perlin_flag    = self.config.noise_perlin_flag,
            gaussian_blur_sigma  = self.config.gaussian_blur_sigma,
            phantom_params       = self.config.phantom_params,
            crop_shape           = self.config.crop_shape,
            initcrop             = self.config.crop_shape,
            jitter               = self.config.jitter,
            max_jitter           = self.config.max_jitter,
            planes               = self.config.planes,
        )

        def crop_center(img, accepted_shape):
            z, y, x = img.shape
            cropz, cropy, cropx = accepted_shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            startz = z // 2 - (cropz // 2)
            return img[startz:startz + cropz, starty:starty + cropy, startx:startx + cropx]

        if data_val == None:
            # generate validation data and store in numpy arrays
            data_val = Data(batch_size=self.config.train_n_val, **data_kwargs)
            data_val = next(data_val.generator())

        elif isinstance(data_val,Data):
            # if the generator defined outside (e.g. in Jupyter) and passed as an argument here
            data_val = next(data_val.generator())

        elif isinstance(data_val,str):
            # load data (some subset of the real data) from a path

            print("str")
            from tifffile import imread

            data_path = Path(data_val)

            files = sorted(data_path.glob('*.tiff'))
            images = {f.stem: imread(str(f)) for f in files}

            # select a 24x subset 0, 6, each +8, -1
            val_images = []
            val_amps = []

            for i, (name, img) in zip(range(176), images.items()):
                if i == 0 or i % 8 == 6 or i == 175:
                    zer, amp = name.split("z")
                    amp = amp.split(".tiff")[0]
                    print(zer, amp)

                    image = normalize(crop_center(img,self.config.crop_shape)) # crop_center(img,accepted_shape)
                    val_images.append(image)
                    val_amps.append([zer,amp])

            X = np.expand_dims(np.stack(val_images, axis=0), -1)
            #Y = np.stack(val_amps, axis=0)
            Y_array = np.zeros((24, 11))
            # to do: add ansi/noll selection. This is for ansi
            zern = np.array([3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            i = 0
            for v in val_amps:
                # print(v, type(v))
                z, a = v
                print(np.where(zern == int(z)))
                Y_array[i, np.where(zern == int(z))] = float(a)
                i += 1

            data_val = X, Y_array

        ## pass prepared data set
        else:
            data_val= data_val[0], data_val[1]


        data_train = Data(batch_size=self.config.train_batch_size, **data_kwargs)

        history = self.keras_model.fit_generator(generator=data_train.generator(), validation_data=data_val,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)
        self._training_finished()
        return history


    def predict(self, img, axes=None, normalizer=None, **predict_kwargs):
        """Predict.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.

        """
        if axes is None:
            axes = self.config.axes
            assert 'C' in axes
            if img.ndim == len(axes)-1 and self.config.n_channel_in == 1:
                # img has no dedicated channel axis, but 'C' always part of config axes
                axes = axes.replace('C','')

        axes     = axes_check_and_normalize(axes,img.ndim)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img) # x has axes_net semantics
        x.shape == tuple(self.get_model_input_shape()) + (self.config.n_channel_in,) or _raise(ValueError())

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        x = normalizer.before(x, axes_net)

        return self.keras_model.predict(x[np.newaxis], **predict_kwargs)[0]


"""
This part of the software is distributed under BSD 3-Clause License 
by Kira Vinogradova.

This software is modified from https://github.com/mpicbg-csbd/phasenet/
Which is distributed under BSD 3-Clause License:

Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""