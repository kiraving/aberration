"""
Modified from https://github.com/mpicbg-csbd/phasenet/
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
See full license at the end of the document
"""

from csbdeep.utils import _raise
import numpy as np
import random
import warnings
from abc import ABC, abstractmethod
import inspect
import tifffile

from phasenet.phantoms import Phantom3D, Points, Sphere



class Images(Phantom3D):

    """
        3D image to be convoleved with the psf
        :param shape: tuple, image shape as (z,y,x)
        :param filepath: list of strings (for multiple phantoms) or a single str
    """

    def __init__(self, shape, filepath):

        super().__init__(shape)
        self.shape=shape

        self.choice = None
        """
        if filepath2 is not None:
            choice_list = [filepath, filepath2]
            self.choice = np.random.choice(choice_list)
            #print("\n self.choice \n : ", self.choice)
            self.image = self.get_image(self.choice)
        else:
            self.image = self.get_image(filepath)

        """
        if isinstance(filepath, list):
            self.choice = np.random.choice(filepath)
            #print("\n Choice from Filepath \n : ", self.choice)
            self.image = self.get_image(self.choice)
        else:
            self.image = self.get_image(filepath)
        #"""

        #print("\nNew Phantom image is initiated \n   ",self.choice)
        self.generate()

    def get_image(self, filep):
        #print("\n Getting Phantom image from \n   ", str(filep))
        return tifffile.imread(filep)

    def generate(self):

        self.image.ndim == 3 or _raise(ValueError("3D image required"))

        self.phantom_obj = self.image
        #print("\nNew Phantom is generated \n   ", self.choice)
        self.check_phantom_obj()
    
    def get(self):
        self.check_phantom_obj()
        return self.phantom_obj

Phantom3D.register(Points)
Phantom3D.register(Sphere)
Phantom3D.register(Images)

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