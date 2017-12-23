import os
import numpy as np


class Decoder(object):
    """
    Viterbi decoder for one-state HMMs.
    """

    def __init__(self, feeder, language_model, bigram=False):
        """
        Initialize decoder.
        :param feeder: (object) feeder object
        :param language_model: (dict) n-gram language model
        :param bigram: (boolean) bigram language model used
        """
        self.network = self._get_network(feeder)
        self.language_model = language_model
        self.bigram = bigram

        self.transition = np.array([hmm["transition"] for hmm in self.network])
        self.selfloop = np.array([hmm["selfloop"] for hmm in self.network])

    def _get_network(self, feeder):
        """
        Build HMMs network for decoding.
        :param feeder: (object) feeder object
        :return: (list) network structure with selfloop and transition probabilities
        """
        leaves_path = os.path.basename(feeder.features_path).split("_")[0] + ".csv"

        # get correct order of HMMs according to OHE
        hmms = feeder.one_hot_decode(np.eye(len(feeder.encoder.classes_), dtype=int))
        hmms_order = {phoneme: i for i, phoneme in enumerate(hmms)}

        # build network structure
        with open(os.path.join("..", "data", "hmm", leaves_path)) as fr:
            network = []
            for i, line in enumerate(fr.readlines()):
                params = line.strip().split(",")
                network.append((hmms_order[params[0]], {
                    "phoneme": params[0],
                    "transition": np.log10(float(params[1])),
                    "selfloop": np.log10(float(params[2]))
                }))

        return [node[1] for node in sorted(network)]

    def _get_penalty(self, min_index=None):
        """
        Get language model penalty.
        :param min_index: (int) index of last visited HMM
        :return: (ndarray) language model penalties per HMM
        """
        fallback = np.log10(1e-7)

        if min_index and self.bigram:
            keys = [(self.network[min_index]["phoneme"], hmm["phoneme"]) for hmm in self.network]
        else:
            keys = [(hmm["phoneme"],) for hmm in self.network]

        return np.array([self.language_model.get(key, fallback) for key in keys])

    def _get_min_likelyhood(self, trellis, min_index):
        """
        Calculate minimum likelyhood.
        :param trellis:
        :param min_index: (int) index of last visited HMM
        :return: (tuple) index and minimum likelyhood
        """
        likelyhood = trellis - self.transition - self._get_penalty(min_index)
        min_index = np.argmin(likelyhood)

        return min_index, likelyhood[min_index]

    def decode(self, observations):
        """
        Find and decode most likely path.
        :param observations: (ndarray) phoneme probabilities per timesteps
        :return: (ndarray) decoded transcription
        """
        observations = np.log10(observations)

        # initialize
        trellis_dim = (len(self.network), len(observations))

        trellis = np.zeros(trellis_dim)
        backpointer = np.ones(trellis_dim).astype(np.int32)
        min_index = None

        trellis[:, 0] = -observations[0] - self._get_penalty()

        # find most likely paths
        for t in range(1, len(observations)):
            previous_trellis = trellis[:, t - 1]

            # last emitting state is the one with lowest likelyhood
            min_index, min_likelyhood = self._get_min_likelyhood(previous_trellis, min_index)

            previous_state = np.repeat(min_likelyhood, trellis_dim[0])
            same_state = trellis[:, t - 1] - self.selfloop

            concatenated_states = np.array([previous_state, same_state])

            trellis[:, t] = np.min(concatenated_states, axis=0) - observations[t]
            backpointer[:, t] = np.argmin(concatenated_states, axis=0)

        # decode best path
        tokens = [trellis[:, -1].argmin()]

        for t in range(len(observations) - 1, 0, -1):
            if backpointer[tokens[-1], t]:
                continue

            tokens.append(trellis[:, t - 1].argmin())

        return [self.network[token]["phoneme"] for token in tokens[::-1]]
