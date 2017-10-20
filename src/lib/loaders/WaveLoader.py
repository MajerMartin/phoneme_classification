import os
import re
import h5py
import codecs
import numpy as np
from glob import glob
from collections import defaultdict
from scipy.io.wavfile import read as wav_read


class WaveLoader(object):
    """
    Load wave files, compute features, create labels and save to file.
    """

    def __init__(self, framer, extractor):
        """
        Initialize wave loader.
        :param framer: (object) instance of Framer object
        :param extractor: (object) instance of object inherited from BaseExtractor
        """
        self.framer = framer
        self.extractor = extractor

    def _rebuild_dir_path(self, path):
        """
        Rebuild path for current operating system.
        :param path: (string) path separated with / or \\
        :return: (string) rebuilt path
        """
        dirs = re.split(r"[\\/]", path)
        return os.path.join(*dirs)

    def _create_full_paths(self, prefix, suffixes):
        """
        Rebuild paths for current operating system and join subdirectories.
        :param prefix: (string) path to subdirectories
        :param suffixes: (list) subdirectories to join
        :return: (list) full paths to subdirectories
        """
        os_prefix = self._rebuild_dir_path(prefix)
        return [os.path.join(os_prefix, suffix) for suffix in suffixes]

    def _build_output_path(self, output_dir_path, input_dir_path):
        """
        Rebuild path for current operating system and join output filename.
        :param output_dir_path: (string) output directory path
        :param input_dir_path: (string) input directory path
        :return: (string) path to output file without extension
        """
        sep = "_"
        ms_to_s = 100

        # prepare path in proper format for current OS
        tmp_path = self._rebuild_dir_path(output_dir_path)

        output_path = os.path.join(tmp_path, sep.join([
            os.path.basename(input_dir_path),
            str(self.framer.frame_length * ms_to_s),
            str(self.framer.frame_step * ms_to_s),
            self.extractor.feature_type,
            "deltas" if self.extractor.deltas else ""
        ]).replace(".", "") + ".").replace(sep + ".", ".")

        return output_path

    def _save_metadata(self, metadata_file_path, regex):
        """
        Save dataset metadata into text file.
        :param metadata_file_path: (string) path to output text file
        :param regex: (object) compiled regular expression
        """
        format_print = lambda k, v: "{:20}|\t{}\n".format(k, v)

        with open(metadata_file_path, "w") as fw:
            for key, value in self.framer.__dict__.iteritems():
                fw.write(format_print(key, value))

            for key, value in self.extractor.__dict__.iteritems():
                fw.write(format_print(key, value))

            fw.write(format_print("regex", regex.pattern))

    def _find_duplicate_files(self, wav_files, regex):
        """
        Find files which share same identification pair (speaker, utterance).
        :param wav_files: (list) path to wave files
        :param regex: (object) compiled regular expression
        :return: (dict) duplicate files per identification pair
        """
        duplicates = {}

        for wav_file in wav_files:
            key = "-".join(regex.findall(os.path.basename(wav_file))[0])

            if key not in duplicates:
                duplicates[key] = [wav_file]
                continue

            duplicates[key].append(wav_file)

        return {key: value for key, value in duplicates.items() if len(value) > 1}

    def _ignore_mlf_condition(self, line):
        """
        Ignore redundant lines in MLF file.
        :param line: (string) one line of MLF file
        :return: (boolean) ignore line
        """
        return line.startswith(("#!MLF!#", ".")) or line == ""

    def _htk_time_to_sec(self, htk_time):
        """
        Convert HTK time to seconds.
        :param htk_time: (str) time in HTK format
        :return: (float) time in seconds
        """
        return float(htk_time) / 10000000

    def _get_words(self, words_path, encoding, regex):
        """
        Collect all words contained in recordings per file per speaker.
        :param words_path: (string) path to file with words
        :param encoding: (string) file encoding
        :param regex: (object) compiled regular expression
        :return: (defaultdict) list of words per file per speaker
        """
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
        """
        Collect all transcriptions and alignments per file per speaker.
        :param alignment_path: (string) path to file with alignments
        :param regex: (object) compiled regular expression
        :param delimiter: (string) delimiter between beginning time, end time and phoneme
        :return: (tuple): (defaultdict) transcriptions per file per speaker
                          (defaultdict) beginning and end time of phonemes per file per speaker
                          (dict) all sorted phonemes
        """
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
                end = self._htk_time_to_sec(end)

                transcriptions[speaker][utterance].append(phoneme)
                alignments[speaker][utterance].append((beginning, end))
                phonemes[phoneme] = True

        return transcriptions, alignments, sorted(phonemes.keys())

    def _process_wavs(self, wav_path, transcriptions, alignments, words, phonemes, regex, output_path):
        """
        Load wave files, compute features, create label and save dataset and its metadata.
        :param wav_path: (string) path to directory containing wave files
        :param transcriptions: (defaultdict) transcriptions per file per speaker
        :param alignments: (defaultdict) beginning and end time of phonemes per file per speaker
        :param words: (defaultdict) list of words per file per speaker
        :param phonemes: (dict) all sorted phonemes
        :param regex: (object) compiled regular expression
        :param output_path: (string) path to output file without extensions
        """
        wav_files = [y for x in os.walk(wav_path) for y in glob(os.path.join(x[0], "*.wav"))]

        # find files with duplicate key pairs (speaker, utterance)
        duplicates = self._find_duplicate_files(wav_files, regex)

        if duplicates:
            for i, d in enumerate(duplicates.itervalues()):
                print "{})".format(i), ", ".join(d)
            raise ValueError("Resolve file names with duplicate (speaker, utterance) pairs.")

        group_sep = "/"

        features_file_path = output_path + "hdf5"
        metadata_file_path = output_path + "txt"

        try:
            os.remove(features_file_path)
            os.remove(metadata_file_path)
        except OSError:
            pass

        with h5py.File(features_file_path, "a") as fw:
            total = len(wav_files)

            for i, wav_file in enumerate(wav_files):
                speaker, utterance = regex.findall(os.path.basename(wav_file))[0]
                transcription = transcriptions.get(speaker, {}).get(utterance, None)

                if transcription:
                    print "\r{}-{} ({}/{})".format(speaker, utterance, i + 1, total),

                    rate, signal = wav_read(wav_file)
                    alignment = alignments[speaker][utterance]

                    max_signal_len = np.floor(alignment[-1][1] * rate).astype(np.int32)

                    signal_cut = signal[:max_signal_len]

                    # create vector with labels per sample
                    labels_per_sample = np.repeat("_", max_signal_len).astype("|S8")

                    for phoneme, times in zip(transcription, alignment):
                        beginning = np.floor(times[0] * rate).astype(np.int32)
                        end = np.floor(times[1] * rate).astype(np.int32)
                        labels_per_sample[beginning:end] = np.repeat(phoneme, end - beginning)

                    # create frames and vector with labels per frame
                    frames, indexes = self.framer.get_frames(signal_cut, rate)

                    # add labels for indexes created by padding
                    last_index = indexes[-1][-1]

                    if last_index >= max_signal_len:
                        last_label = labels_per_sample[-1]
                        labels_per_sample = np.concatenate([labels_per_sample,
                                                            np.repeat(last_label,
                                                                      max_signal_len - last_index + 1).astype("|S8")])

                    labels = self.framer.get_frames_labels(labels_per_sample, indexes)

                    # compute feature per frame
                    features = self.extractor.extract_features(frames, rate)

                    # dump to HDF5 file
                    group_key = group_sep.join([speaker, utterance])
                    features_key = group_sep.join([group_key, "features"])
                    labels_key = group_sep.join([group_key, "labels"])

                    fw[features_key] = features
                    fw[labels_key] = labels

                    # strings in h5py are not that straightforward
                    current_words = words[speaker][utterance]

                    group = fw[group_key]
                    dset = group.create_dataset("words", (len(current_words),), dtype=h5py.special_dtype(vlen=str))
                    dset[:] = current_words

            # save all phonemes for OHE
            group = fw["/"]
            dset = group.create_dataset("phonemes", (len(phonemes),), dtype=h5py.special_dtype(vlen=str))
            dset[:] = phonemes

            # save metadata to text file
            self._save_metadata(metadata_file_path, regex)

    def create_dataset(self, input_dir_path, output_dir_path,
                       wav_dir_name, alignment_file, words_file,
                       regex, encoding, delimiter):
        """
        Collect  words, transcriptions, alignments and phonemes and create dataset.
        :param input_dir_path: (string) input directory path
        :param output_dir_path: (string) output directory path
        :param wav_dir_name: (string) name of directory containing wave files
        :param alignment_file: (string) name of file with alignments
        :param words_file: (string) name of file with words
        :param regex: (string) regular expression for matching speakers and utterances
        :param encoding: (string) words file encoding
        :param delimiter: (string) delimiter between beginning time, end time and phoneme in alignment file
        """
        # prepare paths in proper format for current operating system
        wav_path, alignment_path, words_path = self._create_full_paths(input_dir_path,
                                                                       [wav_dir_name,
                                                                        alignment_file,
                                                                        words_file])

        output_path = self._build_output_path(output_dir_path, input_dir_path)

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

        # collect all transcriptions and phoneme alignments per file per speaker and all phonemes for OHE
        transcriptions, alignments, phonemes = self._get_transcriptions(alignment_path, regex, delimiter)

        # get frames and labels per file per speaker, compute features and dump to HDF5 file
        self._process_wavs(wav_path, transcriptions, alignments, words, phonemes, regex, output_path)

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
