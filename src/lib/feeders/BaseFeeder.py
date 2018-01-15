import os
import re
import h5py
import random
import numpy as np
from sklearn.utils import shuffle as sk_shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


class BaseFeeder(object):
    """
    Split features into train, validation and test set and prepare them for neural network.
    """

    def __init__(self, features_path, noise=None):
        """
        Initialize base feeder.
        :param features_path: (string) path to file with features
        :param noise: (float) standard deviation of Gaussian noise to be added to signal
        """
        self.features_path = self._rebuild_path(features_path)
        self.tmp_storage_path = self.features_path + "_" + str(id(self)) + ".hdf5"

        self.noise = noise

        self.train_speakers = None
        self.val_speakers = None
        self.test_speakers = None

        self.left_context = None
        self.right_context = None
        self.time_steps = None
        self.ctc = False

        self.X_prefix = "X_"
        self.y_prefix = "y_"
        self.bounds_prefix = "bounds_"
        self.transcription_prefix = "transcription_"

        self.encoder = LabelBinarizer()

        self._fit_encoder()

    def _sample_speakers(self, max_train, max_val, max_test):
        """
        Limit number of speakers in train, validation a test sets.
        :param max_train: (int) maximum count of train speakers
        :param max_val: (int) maximum count of validation speakers
        :param max_test: (int) maximum count of test speakers
        """
        self.train_speakers = self.train_speakers[:min(max_train, len(self.train_speakers))] if not None else None
        self.val_speakers = self.val_speakers[:min(max_val, len(self.val_speakers))] if not None else None
        self.test_speakers = self.test_speakers[:min(max_test, len(self.test_speakers))] if not None else None

    def _rebuild_path(self, path):
        """
        Rebuild path for current operating system.
        :param path: (string) path separated with / or \\
        :return: (string) rebuilt path
        """
        dirs = re.split(r"[\\/]", path)
        return os.path.join(*dirs)

    def _fit_encoder(self):
        """
        Fit encoder on all phonemes present in dataset.
        """
        with h5py.File(self.features_path, "r") as fr:
            self.encoder.fit(fr['phonemes'][:])

    def _one_hot_encode(self, labels):
        """
        One-hot encode labels.
        :param labels: (ndarray) labels to transform
        :return: (ndarray) transformed labels
        """
        return self.encoder.transform(labels)

    def _build_features_with_context(self, features, left_context, right_context):
        """
        Pad current frame's features with features from previous and future frames.
        :param features: (ndarray) features for all frames
        :param left_context: (int) number of previous frames
        :param right_context: (int) number of future frames
        :return: (ndarray) padded features for all applicable frames
        """
        self.left_context = left_context
        self.right_context = right_context

        context_count = left_context + right_context
        rows, cols = features.shape

        features_raveled = features.ravel()
        context_features = np.zeros((rows - context_count, cols * (context_count + 1)))

        for i in range(rows - context_count):
            context_features[i, :] = features_raveled[i * cols:(context_count + i + 1) * cols]

        return context_features

    def _process_speakers(self):
        """
        Process speakers - implemented in inherited classes.
        """
        raise NotImplementedError("Implemented in inherited classes.")

    def _train_val_test_split(self, ratio, test_speakers=[]):
        """
        Split dataset into train, validation and test set.
        :param ratio: (tuple) ratio between train, validation and test size
        :param test_speakers: (list) predefined test speakers
        """
        random_state = 42

        # ratio = (ratio between train and val+test, ratio between test and val)
        # train only - (1,0), validation only - (0,0), test only - (0,1)
        train_size, test_size = ratio

        if 0 > train_size > 1 or 0 > test_size > 1:
            raise ValueError("0 <= ratio <= 1.")

        with h5py.File(self.features_path, "r") as fr:
            speakers = sorted([key for key in list(fr.keys()) if key not in ["max_frames_count", "phonemes"] + test_speakers])

        # case when we have predefined test speakers
        if test_speakers:
            self.test_speakers = test_speakers

        # case when we do not want any split
        if train_size == 1:
            self.train_speakers = speakers
            return
        elif test_size == 1 and not test_speakers:
            self.test_speakers = speakers
            return
        elif np.sum(ratio) == 0:
            self.val_speakers = speakers
            return
        elif test_size == 1 and test_speakers:
            return

        # few other cases, ignore the rest
        if train_size > 0 and test_size == 1:
            self.train_speakers, self.test_speakers = train_test_split(speakers,
                                                                       train_size=train_size,
                                                                       random_state=random_state)
        elif train_size > 0 and test_size == 0:
            self.train_speakers, self.val_speakers = train_test_split(speakers,
                                                                      train_size=train_size,
                                                                      random_state=random_state)
        elif train_size > 0 and 0 < test_size < 1 and not test_speakers:
            self.train_speakers, val_test_speakers = train_test_split(speakers,
                                                                      train_size=train_size,
                                                                      random_state=random_state)
            self.val_speakers, self.test_speakers = train_test_split(val_test_speakers,
                                                                     test_size=test_size,
                                                                     random_state=random_state)
        else:
            raise ValueError("Invalid ratio.")

    def one_hot_decode(self, labels):
        """
        Decode one-hot encoded labels.
        :param labels: (ndarray) labels to transform
        :return: (ndarray) transformed labels
        """
        return self.encoder.inverse_transform(labels)

    def create_datasets(self):
        """
        Create datasets - implemented in inherited classes.
        """
        raise NotImplementedError("Implemented in inherited classes.")

    def yield_batches(self, batch_size, split_type, shuffle=True):
        """
        Yield batches of defined batch size.
        :param batch_size: (int) size of batches
        :param split_type: (string) set to yield from (train/val/test)
        :param shuffle: (boolean) shuffle on every epoch
        :return: (ndarray, ndarray) features and labels (generator)
        """
        with h5py.File(self.tmp_storage_path) as fr:
            count = fr[self.y_prefix + split_type][:].shape[0]
            indexes = np.arange(0, count)

            while 1:
                # shuffle in place
                if shuffle:
                    random.shuffle(indexes)

                # yield batches
                for index in range(0, count, batch_size):
                    batch_indexes = sorted(indexes[index:min(index + batch_size, count)])

                    X = fr[self.X_prefix + split_type][batch_indexes, :]
                    y = fr[self.y_prefix + split_type][batch_indexes]

                    if shuffle:
                        X, y = sk_shuffle(X, y)

                    if self.noise and split_type == "train":
                        X += np.random.normal(0, self.noise, X.shape)

                    yield (X, y)

    def remove_tmp_storage(self):
        """
        Remove file with preprocessed features for neural network.
        """
        try:
            os.remove(self.tmp_storage_path)
        except OSError:
            pass

    def get_steps_per_epoch(self, batch_size, split_type):
        """
        Calculate how many batches can be yielded in one epoch without repetition.
        :param batch_size: (int) size of batches
        :param split_type: (string) set to be yielded from (train/val/test)
        :return: (int) number of batches
        """
        with h5py.File(self.tmp_storage_path) as fr:
            return np.ceil(1. * fr[self.y_prefix + split_type].shape[0] / batch_size)

    def split_by_utterance(self, input_array, split_type):
        """
        Split array per corresponding utterances.
        :param input_array: (ndarray) array to split
        :param split_type: (string) set corresponding to input array (train/val/test)
        :return: (ndarray) split array
        """
        with h5py.File(self.tmp_storage_path) as fr:
            bounds = fr[self.bounds_prefix + split_type][:].astype(np.uint32)

        output_array = []
        for bound in bounds:
            output_array.append(input_array[bound[0]:bound[1]])

        return output_array

    def get_dim(self, type, split_type, axis):
        """
        Peek into dataset to get dimension of features or labels.
        :param type: (string) features or labels (X/y)
        :param split_type: (string) set to peek into (train/val/test)
        :param axis: (int) dimension of which axis
        :return: (int) dimension
        """
        prefix = {
            "X": self.X_prefix,
            "y": self.y_prefix,
            "bounds": self.bounds_prefix
        }

        with h5py.File(self.tmp_storage_path) as fr:
            return fr[prefix[type] + split_type].shape[axis]

    def get_transcriptions(self, split_type):
        """
        Yield transcriptions for utterances in set (generator).
        :param split_type: (string) set to get transcriptions for (train/val/test)
        :return: (ndarray) transcription
        """
        with h5py.File(self.features_path, "r") as fr_features:
            with h5py.File(self.tmp_storage_path, "r") as fr_storage:
                for (speaker, utterance) in fr_storage[self.transcription_prefix + split_type][:]:
                    yield fr_features[speaker][utterance]["transcription"][:]

    def __str__(self):
        """
        Override.
        :return: (string) class representation
        """
        output = ""

        for key, value in self.__dict__.items():
            output += "{{'{0}': {1}}}, ".format(key, value)

        output += "{{'{0}': {1}}}".format("encoder.classes_", self.encoder.classes_)

        return output

    def __repr__(self):
        """
        Override.
        :return: (string) class representation
        """
        return self.__str__()
