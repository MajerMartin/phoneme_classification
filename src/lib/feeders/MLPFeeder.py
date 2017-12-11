import h5py
import numpy as np
from BaseFeeder import BaseFeeder


class MLPFeeder(BaseFeeder):
    """
    Split features into train, validation and test set and prepare them for feed-forward neural network.
    """

    def __init__(self, features_path):
        """
        Initialize MLP feeder.
        :param features_path: (string) path to file with features
        """
        super(MLPFeeder, self).__init__(features_path)

    def _process_speakers(self, speakers, suffix, left_context, right_context, fr, fw):
        """
        Create train, validation or test set for list of speakers.
        :param speakers: (list) speakers to process
        :param suffix: (string) identifier of split
        :param left_context: (int) number of previous frames
        :param right_context: (int) number of future frames
        :param fr: (object) file read object
        :param fw: (object) file write object
        """
        # find dimension of resulting datasets
        max_rows = 0
        max_cols_features = 0
        max_cols_labels = 0
        utterances_count = 0

        context_count = left_context + right_context

        for speaker in speakers:
            for i, utterance in enumerate(fr[speaker].keys()):
                features_data = fr[speaker][utterance]["features"]
                labels_data = fr[speaker][utterance]["labels"]

                if i == 0:
                    max_cols_features = features_data.shape[1] + features_data.shape[1] * context_count
                    max_cols_labels = self._one_hot_encode(labels_data[:2]).shape[1]

                max_rows += features_data.shape[0] - context_count
                utterances_count += 1

        # create datasets
        X = fw.create_dataset(self.X_prefix + suffix, (max_rows, max_cols_features))
        y = fw.create_dataset(self.y_prefix + suffix, (max_rows, max_cols_labels))
        bounds = fw.create_dataset(self.bounds_prefix + suffix, (utterances_count, 2))
        transcription_map = fw.create_dataset(self.transcription_prefix + suffix, (utterances_count, 2),
                                              dtype=h5py.special_dtype(vlen=str))

        # process features and labels and store them in datasets
        rows_count = 0
        utterance_index = 0
        speakers_count = len(speakers)

        print suffix

        for i, speaker in enumerate(speakers):
            print "\r\t({}/{})".format(i + 1, speakers_count),

            for utterance in fr[speaker].keys():
                features = fr[speaker][utterance]["features"][:]
                labels = fr[speaker][utterance]["labels"][:]

                if left_context or right_context:
                    features = self._build_features_with_context(features, left_context, right_context)
                    labels = labels[left_context:-right_context]

                labels_ohe = self._one_hot_encode(labels)

                current_rows_count = rows_count
                rows_count += labels.shape[0]

                X[current_rows_count:rows_count, :] = features
                y[current_rows_count:rows_count] = labels_ohe
                bounds[utterance_index, :] = np.array([current_rows_count, rows_count])
                transcription_map[utterance_index, :] = np.array([speaker, utterance])

                utterance_index += 1

        print "\n\t{{'{0}': {1}}}, {{'{2}': {3}}}".format("X.shape", X.shape, "y.shape", y.shape)

    def create_datasets(self, ratio, test_speakers=[], left_context=0, right_context=0, sample=[]):
        """
        Create train, validation and test datasets.
        :param ratio: (tuple) ratio between train, validation and test size
        :param test_speakers: (list) predefined test speakers
        :param left_context: (int) number of previous frames
        :param right_context: (int) number of future frames
        :param sample: (list) train, validation and test size limits
        """
        self._train_val_test_split(ratio, test_speakers)

        if sample:
            self._sample_speakers(*sample)

        with h5py.File(self.features_path, "r") as fr:
            with h5py.File(self.tmp_storage_path, "a") as fw:
                if self.train_speakers:
                    self._process_speakers(self.train_speakers, "train", left_context, right_context, fr, fw)

                if self.val_speakers:
                    self._process_speakers(self.val_speakers, "val", left_context, right_context, fr, fw)

                if self.test_speakers:
                    self._process_speakers(self.test_speakers, "test", left_context, right_context, fr, fw)
