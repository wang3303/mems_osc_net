from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import cmath
from mpl_toolkits.mplot3d import Axes3D

from numpy.core.umath_tests import inner1d
import math
from scipy.sparse import issparse

def bin2sign(matrix):
    return np.where(matrix == 0, -1, 1)

def hopfield_energy(weight, input_data, output_data):
    return -0.5 * inner1d(input_data.dot(weight), output_data)

def binarize(input_value):
    return np.where(input_value > 9, 1, 0)

def format_data(data, is_feature1d=True, copy=False):
    if data is None or issparse(data):
        return data
    if not isinstance(data, np.ndarray) or copy:
        data = np.array(data, copy=copy)
    n_features = data.shape[-1]
    if data.ndim == 1:
        data_shape = (n_features, 1) if is_feature1d else (1, n_features)
        data = data.reshape(data_shape)
    return data

import warnings

class HopfieldNetwork():
    def __init__(self,):
        self.n_memorized_samples = 0
        self.weight=None
        self.limit = 10

    # Validate the input 
    def discrete_validation(self, matrix):
        if np.any((matrix != 0) & (matrix != 1)):
            raise ValueError("Data should contain 0 and 1 values")    
    
    def train(self, input_data):
        self.discrete_validation(input_data)

        input_data = bin2sign(input_data)
        input_data = format_data(input_data, is_feature1d=False)

        n_rows, n_features = input_data.shape
        n_rows_after_update = self.n_memorized_samples + n_rows
        
        if n_rows_after_update > self.limit:
            warnings.warn("Number of remebered pattern exceeds limit.")
        
        weight_shape = (n_features, n_features)

        if self.weight is None:
            self.weight = np.zeros(weight_shape, dtype=int)

        if self.weight.shape != weight_shape:
            n_features_expected = self.weight.shape[1]
            raise ValueError("Input data has invalid number of features. "
                             "Got {} features instead of {}."
                             "".format(n_features, n_features_expected))

        self.weight = input_data.T.dot(input_data)
        np.fill_diagonal(self.weight, np.zeros(len(self.weight)))
        self.n_memorized_samples = n_rows_after_update

    def predict(self, input_data, n_times=None):
        self.discrete_validation(input_data)
        input_data = format_data(bin2sign(input_data), is_feature1d=False)
        output_data = input_data.dot(self.weight)
        return binarize(output_data).astype(int)

    def energy(self, input_data):
        self.discrete_validation(input_data)
        input_data = bin2sign(input_data)
        input_data = format_data(input_data, is_feature1d=False)
        n_rows, n_features = input_data.shape

        if n_rows == 1:
            return hopfield_energy(self.weight, input_data, input_data)

        output = np.zeros(n_rows)
        for i, row in enumerate(input_data):
            output[i] = hopfield_energy(self.weight, row, row)
        return output