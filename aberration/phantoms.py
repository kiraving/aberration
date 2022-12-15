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