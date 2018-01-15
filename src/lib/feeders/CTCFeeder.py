import h5py
import random
import numpy as np
from itertools import groupby, chain
from sklearn.utils import shuffle as sk_shuffle
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
        self.input_length_prefix = "input_length"
        self.label_length_prefix = "label_length"
        self.ctc = True

        with h5py.File(self.features_path, "r") as fr:
            self.max_sequence_length = fr["max_frames_count"][:][0]

            # remap phonemes to numeric values with blank label (-) as a last one
            self.phonemes_map = {phoneme: i for i, phoneme in
                                 enumerate(np.concatenate([fr["phonemes"][:], self.blank]))}
            self.inverse_phonemes_map = {value: key for key, value in self.phonemes_map.items()}

    def _set_max_labelling_length(self, fr):
        """
        Set maximum labelling length.
        :param fr: (object) file read object
        """
        for speaker in list(chain(*[self.train_speakers, self.val_speakers, self.test_speakers])):
            for utterance in fr[speaker].keys():
                self.max_labelling_length = max(self.max_labelling_length,
                                                fr[speaker][utterance]["transcription"][:].shape[0])

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
        if self.max_labelling_length == 0:
            self._set_max_labelling_length(fr)

        # find dimension of resulting datasets
        max_cols_features = 0
        utterances_count = 0

        context_count = left_context + right_context
        self.max_sequence_length -= context_count

        for speaker in speakers:
            for i, utterance in enumerate(fr[speaker].keys()):
                features_data = fr[speaker][utterance]["features"]

                if i == 0:
                    max_cols_features = features_data.shape[1] + features_data.shape[1] * context_count

                utterances_count += 1

        # create datasets
        X = fw.create_dataset(self.X_prefix + suffix, (utterances_count, self.max_sequence_length, max_cols_features))
        y = fw.create_dataset(self.y_prefix + suffix, (utterances_count, self.max_labelling_length))
        input_length = fw.create_dataset(self.input_length_prefix + suffix, (utterances_count,))
        label_length = fw.create_dataset(self.label_length_prefix + suffix, (utterances_count,))
        transcription_map = fw.create_dataset(self.transcription_prefix + suffix, (utterances_count, 2),
                                              dtype=h5py.special_dtype(vlen=str))

        # process features and labels and store them in datasets
        utterance_index = 0
        speakers_count = len(speakers)

        print(suffix)

        for i, speaker in enumerate(speakers):
            print("\r\t({}/{})".format(i + 1, speakers_count), end=" ")

            for utterance in list(fr[speaker].keys()):
                features = fr[speaker][utterance]["features"][:]
                labels = fr[speaker][utterance]["labels"][:]

                if left_context or right_context:
                    features = self._build_features_with_context(features, left_context, right_context)
                    if not right_context:
                        labels = labels[left_context:]
                    else:
                        labels = labels[left_context:-right_context]

                # encode labels to numeric value
                labels = [self.phonemes_map[label.decode()] for label in labels]

                # pad sequence with zeros
                padding = np.zeros((self.max_sequence_length - features.shape[0], features.shape[1]))
                features = np.vstack([padding, features])

                # get correct label
                labels = np.array([k for k, g in groupby(labels)])

                input_length_data = np.array([len(features)])
                label_length_data = np.array([len(labels)])

                # pad label with minus one
                padding = np.ones((self.max_labelling_length - labels.shape[0],)) * -1
                labels = np.hstack([labels, padding])

                features = features.reshape(1, features.shape[0], features.shape[1])
                labels = labels.reshape(1, -1)

                X[utterance_index, :, :] = features
                y[utterance_index, :] = labels
                input_length[utterance_index] = input_length_data
                label_length[utterance_index] = label_length_data
                transcription_map[utterance_index, :] = np.array([speaker, utterance])

                utterance_index += 1
        print("\n\t{{'{0}': {1}}}, {{'{2}': {3}}}".format("X.shape", X.shape, "y.shape", y.shape))

    def yield_batches(self, batch_size, split_type, shuffle=True):
        """
        Yield batches of defined batch size.
        :param batch_size: (int) size of batches
        :param split_type: (string) set to yield from (train/val/test)
        :param shuffle: (boolean) shuffle on every epoch
        :return: (dict, dict) features, labels and other information and dummy output (generator)
        """
        with h5py.File(self.tmp_storage_path) as fr:
            count = fr[self.y_prefix + split_type][:].shape[0]
            indexes = np.arange(0, count)

            if shuffle:
                random.shuffle(indexes)

            # yield batches
            for index in range(0, count, batch_size):
                batch_indexes = sorted(indexes[index:min(index + batch_size, count)])

                X = fr[self.X_prefix + split_type][batch_indexes, :, :]
                y = fr[self.y_prefix + split_type][batch_indexes, :]
                input_length = fr[self.input_length_prefix + split_type][batch_indexes]
                label_length = fr[self.label_length_prefix + split_type][batch_indexes]

                if shuffle:
                    X, y, input_length, label_length = sk_shuffle(X, y, input_length, label_length)

                if self.noise and split_type == "train":
                    X += np.random.normal(0, self.noise, X.shape)

                inputs = {
                    "the_input": X,
                    "the_labels": y,
                    "input_length": input_length,
                    "label_length": label_length
                }

                # dummy data for dummy loss function
                outputs = {"ctc": np.zeros([X.shape[0]])}

                yield (inputs, outputs)
