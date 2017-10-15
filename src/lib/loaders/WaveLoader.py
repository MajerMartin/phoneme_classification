import os
import re
import h5py
import codecs
import numpy as np
from glob import glob
from collections import defaultdict
from scipy.io.wavfile import read as wav_read


class WaveLoader(object):
    def __init__(self, framer, extractor):
        self.framer = framer
        self.extractor = extractor

    def _rebuild_dir_path(self, path):
        dirs = re.split(r"[\\/]", path)
        return os.path.join(*dirs)

    def _create_full_paths(self, prefix, sufixes):
        os_prefix = self._rebuild_dir_path(prefix)
        return [os.path.join(os_prefix, sufix) for sufix in sufixes]

    def _ignore_mlf_condition(self, line):
        return line.startswith(("#!MLF!#", ".")) or line == ""

    def _htk_time_to_sec(self, htk_time):
        return float(htk_time) / 10000000

    def _slice_signal(self, rate, data, beginning, end):
        start = np.floor(beginning * rate).astype(np.int32)
        stop = np.floor(end * rate).astype(np.int32)
        return data[start:stop]

    def _get_words(self, words_path, encoding, regex):
        # this file should not be that large - load it into memory
        with codecs.open(words_path, mode="rt", encoding=encoding) as fr:
            lines = fr.read().split("\n")

        words = defaultdict(dict)

        for line in lines:
            line = line.strip()

            if self._ignore_mlf_condition(line):
                continue

            if ".lab" in line:
                speaker, utterance = regex.findall(line)[0]
                words[speaker][utterance] = []
                continue

            words[speaker][utterance].append(line)

        return words

    def _get_transcriptions(self, alignment_path, regex, delimiter):
        transcriptions = defaultdict(dict)
        alignments = defaultdict(dict)
        phonemes = {}

        with open(alignment_path, 'r') as fr:
            # this file may be large - do not load into memory and read line by line instead
            for line in fr:
                line = line.strip()

                if self._ignore_mlf_condition(line):
                    continue

                if ".lab" in line:
                    speaker, utterance = regex.findall(line)[0]
                    transcriptions[speaker][utterance] = []
                    alignments[speaker][utterance] = []
                    continue

                # keep first three elements and discard the rest
                beginning, end, phoneme = line.split(delimiter)[0:3]

                beginning = self._htk_time_to_sec(beginning)
                end = self._htk_time_to_sec(beginning)

                transcriptions[speaker][utterance].append(phoneme)
                alignments[speaker][utterance].append((beginning, end))
                phonemes[phoneme] = True

        return transcriptions, alignments, sorted(phonemes.keys())

    def _process_wavs(self, wav_path, transcriptions, alignments, regex):
        pass

    def create_dataset(self, input_dir_path, output_dir_path,
                       wav_dir_name, alignment_file, words_file,
                       regex, encoding, delimiter):
        # prepare paths in proper format for current OS
        wav_path, alignment_path, words_path = self._create_full_paths(input_dir_path,
                                                                       [wav_dir_name,
                                                                        alignment_file,
                                                                        words_file])
        output_path = self._rebuild_dir_path(output_dir_path)

        # select appropriate regex - either by selecting known dataset or using custom one
        known_datasets = {
            "SkodaAuto": ".*(spk\-\d*)\-(\d*).*",
            "SpeechDat-E": ".*([A-Z]\d*)([A-Z]\d*).*"
        }

        if regex in known_datasets:
            regex = known_datasets[regex]

        regex = re.compile(regex)

        # collect all words per file per speaker
        words = self._get_words(words_path, encoding, regex)

        # collect all transcriptions and phone alignments per file per speaker and all phonemes for OHE
        transcriptions, alignments, phonemes = self._get_transcriptions(alignment_path, regex, delimiter)

        # get frames and labels per file per speaker and compute features
        features, labels = self._process_wavs(wav_path, transcriptions, alignments, regex)

        # dump all data and metadata into HDF5 file
        # TODO
        # when dumping metadata, add print self.framer, print self.extractor, data type dir, data regex
        # and possibly self.extractor.feature_type
        # build filename from metadata -
        # save features, words, labels, transcriptions per speaker per file, phonemes

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