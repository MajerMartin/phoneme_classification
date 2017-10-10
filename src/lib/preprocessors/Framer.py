import numpy as np
from collections import Counter


class Framer(object):
    """
    Ingest signal, split it into frames and create label for every frame.
    """

    def __init__(self, frame_length, frame_step, window_function=np.hamming):
        """
        Initialize framer.
        :param frame_length: (float) length of single frame in seconds
        :param frame_step: (float) length of frame step in seconds
        :param window_function: (function) window function to be applied to every frame
        """
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.window_function = window_function

    def get_frames(self, signal, rate):
        """
        Split signal into frames (with possible overlap) and apply window function.
        :param signal: (ndarray) 1D signal
        :param rate: (int) signal frame rate
        :return: (ndarray, ndarray) frames and corresponding signal indexes
        """
        signal_length = len(signal)

        # convert frame length and step size to samples per frame
        frame_length = int(rate * self.frame_length)
        frame_step = int(rate * self.frame_step)

        # calculate frame count
        if signal_length < frame_length:
            frames_count = 1
        else:
            frames_count = 1 + int(np.ceil((signal_length * 1.0 - frame_length) / frame_step))

        # if last frame is incomplete, add padding of zeroes
        padding_length = int((frames_count - 1) * frame_step + frame_length)
        padding = np.zeros((padding_length - signal_length))
        signal = np.concatenate((signal, padding))

        # create array with frame indexes
        indexes = np.tile(np.arange(0, frame_length), (frames_count, 1)) + np.tile(
            np.arange(0, frames_count * frame_step, frame_step), (frame_length, 1)).T

        # use indexes mask to get single frames
        frames = signal[indexes]

        # create window function mask
        windows = np.tile(self.window_function(frame_length), (frames_count, 1))

        return frames * windows, indexes

    def get_frames_labels(self, labels_per_sample, indexes):
        """
        Create labels for every frame.
        :param labels_per_sample: (ndarray) labels per sample in signal
        :param indexes: (ndarray) indexes corresponding to frames
        :return: (list) labels per frames
        """
        labels_per_frame = labels_per_sample[indexes]
        labels = [Counter(lbs).most_common()[0][0] for lbs in labels_per_frame]

        return labels

    def __str__(self):
        """
        Override.
        :return: (string) class representation
        """
        output = ""

        for key, value in self.__dict__.iteritems():
            output += "{{'{0}': {1}}}, ".format(key, value)

        return output[:-2]

    def __repr__(self):
        """
        Override.
        :return: (string) class representation
        """
        return self.__str__()
