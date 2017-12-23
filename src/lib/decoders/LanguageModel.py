import numpy as np
from collections import Counter


class LanguageModel(object):
    """
    Build n-gram language model from train set transcriptions.
    """

    def __init__(self, feeder):
        """
        Initialize language model.
        :param feeder: (object) feeder object
        """
        self.feeder = feeder

    def _get_ngrams(self, n=2):
        """
        Create n-grams from train set transcriptions and get their frequencies.
        :param n: (int) gram order
        :return: (dict) counter of n-grams
        """
        cnt = Counter()

        for transcription in self.feeder.get_transcriptions("train"):
            for i in range(len(transcription) - n + 1):
                key = tuple([phoneme for phoneme in transcription[i:i + n]])
                cnt[key] += 1

        return cnt

    def _get_zerogram(self):
        """
        Build zerogram language model.
        :return: (dict) zerogram language model
        """
        phonemes = self.feeder.encoder.classes_
        probabilities = {}

        for phoneme in phonemes:
            probabilities[(phoneme,)] = np.log10(1.0 / len(phonemes))

        return probabilities

    def _get_unigram(self):
        """
        Build unigram language model.
        :return: (dict) unigram language model
        """
        unigrams = self._get_ngrams(n=1)
        probabilities = {}

        for key in unigrams:
            # TODO: implement smoothing
            probabilities[key] = np.log10(unigrams[key] * 1.0 / sum(unigrams.values()))

        return probabilities

    def _get_bigram(self):
        """
        Build bigram language model.
        :return: (dict) bigram language model
        """
        unigrams = self._get_ngrams(n=1)
        bigrams = self._get_ngrams(n=2)
        probabilities = {}

        for key in bigrams:
            # TODO: implement smoothing
            probabilities[key] = np.log10(bigrams[key] * 1.0 / unigrams[(key[0],)])

        return probabilities

    def create_model(self, ngram=1):
        """
        Build n-gram language model.
        :param ngram: (int) gram order
        :return: (dict) n-gram language model
        """
        if ngram == 0:
            return self._get_zerogram()

        if ngram == 1:
            return self._get_unigram()

        if ngram == 2:
            unigram_probabilities = self._get_unigram()
            bigram_probabilities = self._get_bigram()

            merged_probabilities = unigram_probabilities.copy()
            merged_probabilities.update(bigram_probabilities)

            return merged_probabilities
