import numpy as np


class BaseExtractor(object):
    """
    Implements interface for feature extraction, delta and delta-delta coefficients computation and normalization.
    """

    def __init__(self, normalize=True, deltas=False, **kwargs):
        """
        Initialize extractor.
        :param normalize: (boolean) normalize features column-wise
        :param deltas: (boolean) compute delta and delta-delta coefficients
        :param kwargs: (dict) keyword arguments passed to inherited classes
        """
        super(BaseExtractor, self).__init__(**kwargs)
        self.normalize = normalize
        self.deltas = deltas

    def _normalize(self, x):
        """
        Normalize vector to zero mean and unit variance (Z-score normalization).
        :param x: (ndarray) vector to normalize
        :return: (ndarray) normalized vector
        """
        return (x - x.mean()) / x.std()

    def _normalize_columns(self, matrix):
        """
        Normalize matrix column-wise.
        :param matrix: (ndarray) matrix to normalize
        :return: (ndarray) normalized matrix
        """
        dim = matrix.shape

        normalized = np.zeros(dim)

        for i in range(dim[1]):
            column = matrix[:, i]
            normalized[:, i] = self._normalize(column)

        return normalized

    def _compute_deltas(self, matrix, order=1):
        """
        Compute n-th order delta coefficients. Padded with zeros.
        :param matrix: (ndarray) matrix where rows correspond to frames
        :param order: (int) order of discrete difference
        :return: (ndarray) delta coefficients matrix with same shape as input matrix
        """
        axis = 0

        deltas = np.diff(matrix, n=order, axis=axis)

        padding = [(0, 0)] * matrix.ndim
        padding[axis] = (order, 0)
        deltas = np.pad(deltas, padding, mode="constant")

        return deltas

    def _extract_features(self, frames, rate):
        """
        Extract features for every frame - implemented in inherited classes.
        :param frames: (ndarray) signal split into frames
        :param rate: (int) frame rate of audio signal
        :return: (ndarray) features per frame
        """
        raise NotImplementedError("Implemented in inherited classes.")

    def extract_features(self, frames, rate):
        """
        Extract features, possibly calculate delta and delta-delta coefficients and normalize.
        :param frames: (ndarray) signal split into frames
        :param rate: (int) frame rate of audio signal
        :return: (ndarray) features
        """
        features = self._extract_features(frames, rate)

        dim = features.shape

        if len(dim) == 1:
            features = features.reshape(dim[0], 1)
        elif dim[0] == 1:
            features = features.reshape(dim[1], 1)

        if self.deltas:
            deltas = self._compute_deltas(features, order=1)
            delta_deltas = self._compute_deltas(features, order=2)
            features = np.hstack([features, deltas, delta_deltas])

        if self.normalize:
            features = self._normalize_columns(features)

        return features

    def __str__(self):
        """
        Override.
        :return: (string) class representation
        """
        output = ""

        for key, value in self.__dict__.items():
            if type(value) == str:
                value = "'{}'".format(value)
            output += "{{'{0}': {1}}}, ".format(key, value)

        return output[:-2]

    def __repr__(self):
        """
        Override.
        :return: (string) class representation
        """
        return self.__str__()
