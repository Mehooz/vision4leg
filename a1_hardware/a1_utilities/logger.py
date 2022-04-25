import numpy as np
import cv2
import os
import time
import pathlib
from collections import deque
import threading
import os


class VisualLogger():
    def __init__(
        self,
        save_path
    ):        
        self.idx = 0
        self.data_save_path = save_path
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def record(self, data):
        self.idx += 1
        cv2.imwrite(
            "{}_{}".format(self.idx, time.time()), data
        )


class StateLogger():
    def __init__(self, data_example, duration, frequency, data_save_name):
        self.dtype = data_example.dtype
        self.maxlength = int(duration * frequency * 2.0)

        self.data_i = 0

        self.shape = data_example.shape + (self.maxlength,)

        self.timestamp = np.zeros(self.maxlength, dtype=np.float64)
        self.data = np.zeros(self.shape, self.dtype)

        self.data_save_name = data_save_name
       
    def record(self, data):
        self.timestamp[self.data_i] = time.time()
        self.data[...,self.data_i] = data
        self.data_i += 1

    def write(self):
        np.savez(self.data_save_name, self.timestamp[...,0:self.data_i], self.data[...,0:self.data_i])
        print("Saved as %s" % self.data_save_name)

