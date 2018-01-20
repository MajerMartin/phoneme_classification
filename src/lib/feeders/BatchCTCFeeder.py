import h5py
import numpy as np
from itertools import groupby
from .CTCFeeder import CTCFeeder


class BatchCTCFeeder(CTCFeeder):
    """
    Split features into train, validation and test set into batches per utterance and prepare them
    for recurrent neural network with CTC loss.
    """

    def __init__(self, features_path, noise=None, sequence_length=20, overlap=True):
        """
        Initialize batch CTC feeder.
        :param features_path: (string) path to file with features
        :param noise: (float) standard deviation of Gaussian noise to be added to signal
        :param sequence_length: (int) number of frames in sequence
        :param overlap: (boolean) 50% overlap between train sequences
        """
        super(BatchCTCFeeder, self).__init__(features_path, noise)

        self.sequence_length = sequence_length
        self.max_labelling_length = sequence_length
        self.overlap = 0 if not overlap else sequence_length // 2

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
        overlap_cond = suffix in ["train", "val"] and self.overlap > 0

        for speaker in speakers:
            for i, utterance in enumerate(fr[speaker].keys()):
                features_data = fr[speaker][utterance]["features"]

                if i == 0:
                    max_cols_features = features_data.shape[1] + features_data.shape[1] * context_count

                sequences_count = np.ceil((features_data.shape[0] - context_count) / self.sequence_length)

                if overlap_cond:
                    max_rows += 2 * sequences_count - 1
                else:
                    max_rows += sequences_count

                utterances_count += 1

        # create datasets
        X = fw.create_dataset(self.X_prefix + suffix, (max_rows, self.sequence_length, max_cols_features))
        y = fw.create_dataset(self.y_prefix + suffix, (max_rows, self.sequence_length))
        input_length = fw.create_dataset(self.input_length_prefix + suffix, (max_rows,))
        label_length = fw.create_dataset(self.label_length_prefix + suffix, (max_rows,))
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
                features_tmp = fr[speaker][utterance]["features"][:]
                labels_tmp = fr[speaker][utterance]["labels"][:]

                if left_context or right_context:
                    features_tmp = self._build_features_with_context(features_tmp, left_context, right_context)
                    if not right_context:
                        labels_tmp = labels_tmp[left_context:]
                    else:
                        labels_tmp = labels_tmp[left_context:-right_context]

                # encode labels to numeric value
                labels_tmp = [self.phonemes_map[label.decode()] for label in labels_tmp]

                # pad sequence with zeros
                missing_frames = (
                    np.ceil(features_tmp.shape[0] / self.sequence_length) * self.sequence_length - features_tmp.shape[
                        0])

                padding = np.zeros((missing_frames.astype(int), features_tmp.shape[1]))
                features_tmp = np.vstack([padding, features_tmp])

                # temporarily pad labels
                padding = np.ones((missing_frames.astype(int),)) * -1
                labels_tmp = np.hstack([padding, labels_tmp])

                # reshape to desired shape - quick and dirty approach, in no way optimized
                if overlap_cond:
                    batches = 2 * features_tmp.shape[0] // self.sequence_length - 1

                    features = np.zeros((batches, self.sequence_length, features_tmp.shape[1]))
                    labels_tmp_tmp = np.zeros((batches, self.sequence_length))

                    for j in range(batches):
                        index_from = j * self.overlap
                        index_to = index_from + self.sequence_length

                        features[j, :] = features_tmp[index_from:index_to, :]
                        labels_tmp_tmp[j] = labels_tmp[index_from:index_to]

                    labels_tmp = labels_tmp_tmp
                else:
                    features = features_tmp.reshape(-1, self.sequence_length, features_tmp.shape[1])
                    labels_tmp = labels_tmp.reshape(-1, self.sequence_length)

                # fix labels
                labels = np.zeros(labels_tmp.shape)

                for j, label in enumerate(labels_tmp):
                    # get correct label
                    grouped_label = np.array([k for k, g in groupby(label) if k != -1])

                    # pad label with minus one
                    padding = np.ones((self.sequence_length - len(grouped_label),)) * -1

                    labels[j, :] = np.hstack([grouped_label, padding])

                current_rows_count = rows_count
                rows_count += labels.shape[0]

                X[current_rows_count:rows_count, :, :] = features
                y[current_rows_count:rows_count, :] = labels
                input_length[current_rows_count:rows_count] = np.ones((labels.shape[0],)) * self.sequence_length
                label_length[current_rows_count:rows_count] = [np.sum(label > -1) for label in labels]
                bounds[utterance_index, :] = np.array([current_rows_count, rows_count])
                transcription_map[utterance_index, :] = np.array([speaker, utterance])

                utterance_index += 1
        print("\n\t{{'{0}': {1}}}, {{'{2}': {3}}}".format("X.shape", X.shape, "y.shape", y.shape))
