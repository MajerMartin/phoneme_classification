import h5py
import random
import numpy as np
from itertools import groupby
from .MLPFeeder import MLPFeeder


class CTCFeeder(MLPFeeder):
    """
    Split features into train, validation and test set and prepare them for recurrent neural network with CTC loss.
    """

    def __init__(self, features_path, noise=None):
        """
        Initialize CTC feeder.
        :param features_path: (string) path to file with features
        :param noise: (float) standard deviation of Gaussian noise to be added to signal
        """
        super(CTCFeeder, self).__init__(features_path, noise)

        self.max_labelling_length = 0
        self.blank = np.array(["-"])
        self.ctc = True

        with h5py.File(self.features_path, "r") as fr:
            self.max_sequence_length = fr["max_frames_count"][:][0]

            # remap phonemes to numeric values with blank label (-) as a last one
            self.phonemes_map = {phoneme: i for i, phoneme in
                                 enumerate(np.concatenate([fr['phonemes'][:], self.blank]))}
            self.inverse_phonemes_map = {value: key for key, value in self.phonemes_map.items()}

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
        utterances_count = 0

        context_count = left_context + right_context

        for speaker in speakers:
            for i, utterance in enumerate(fr[speaker].keys()):
                features_data = fr[speaker][utterance]["features"]

                self.max_labelling_length = max(self.max_labelling_length,
                                                fr[speaker][utterance]["transcription"][:].shape[0])
                if i == 0:
                    max_cols_features = features_data.shape[1] + features_data.shape[1] * context_count

                max_rows += features_data.shape[0] - context_count
                utterances_count += 1

        # create datasets
        X = fw.create_dataset(self.X_prefix + suffix, (max_rows, max_cols_features))
        y = fw.create_dataset(self.y_prefix + suffix, (max_rows,))
        bounds = fw.create_dataset(self.bounds_prefix + suffix, (utterances_count, 2))
        transcription_map = fw.create_dataset(self.transcription_prefix + suffix, (utterances_count, 2),
                                              dtype=h5py.special_dtype(vlen=str))

        # process features and labels and store them in datasets
        rows_count = 0
        utterance_index = 0
        speakers_count = len(speakers)

        print(suffix)

        for i, speaker in enumerate(speakers):
            print("\r\t({}/{})".format(i + 1, speakers_count), end=" ")

            for utterance in list(fr[speaker].keys()):
                features = fr[speaker][utterance]["features"][:]
                labels = fr[speaker][utterance]["labels"][:]

                if self.noise and suffix == "train":
                    features += np.random.normal(0, self.noise, features.shape)

                if left_context or right_context:
                    features = self._build_features_with_context(features, left_context, right_context)
                    if not right_context:
                        labels = labels[left_context:]
                    else:
                        labels = labels[left_context:-right_context]

                current_rows_count = rows_count
                rows_count += labels.shape[0]

                X[current_rows_count:rows_count, :] = features
                y[current_rows_count:rows_count] = [self.phonemes_map[label.decode()] for label in labels]
                bounds[utterance_index, :] = np.array([current_rows_count, rows_count])
                transcription_map[utterance_index, :] = np.array([speaker, utterance])

                utterance_index += 1

        print("\n\t{{'{0}': {1}}}, {{'{2}': {3}}}".format("X.shape", X.shape, "y.shape", y.shape))

    def yield_batches(self, split_type, shuffle=True):
        """
        Yield batches of batch size 1.
        :param split_type: (string) set to yield from (train/val/test)
        :param shuffle: (boolean) shuffle on every epoch
        :return: (dict, dict) features, labels and other information and dummy output (generator)
        """
        with h5py.File(self.tmp_storage_path) as fr:
            count = fr[self.bounds_prefix + split_type][:].shape[0]
            indexes = np.arange(0, count)

            if shuffle:
                random.shuffle(indexes)

            for index in indexes:
                bounds = fr[self.bounds_prefix + split_type][index].astype(int)

                X = fr[self.X_prefix + split_type][bounds[0]:bounds[1]]
                y = fr[self.y_prefix + split_type][bounds[0]:bounds[1]]

                # pad sequence with zeros
                padding = np.zeros((self.max_sequence_length - X.shape[0], X.shape[1]))
                X = np.vstack([padding, X])

                # get correct label
                y = np.array([k for k, g in groupby(y)])

                input_length = np.array([len(X)])
                label_length = np.array([len(y)])

                # pad label with minus one
                padding = np.ones((self.max_labelling_length - y.shape[0],)) * -1
                y = np.hstack([y, padding])

                X = X.reshape(1, X.shape[0], X.shape[1])
                y = y.reshape(1, -1)

                inputs = {
                    "the_input": X,
                    "the_labels": y,
                    "input_length": input_length,
                    "label_length": label_length
                }

                # dummy data for dummy loss function
                outputs = {"ctc": np.zeros([X.shape[0]])}

                yield (inputs, outputs)
