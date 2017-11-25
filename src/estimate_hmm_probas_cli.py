import os
import re
import argparse
import numpy as np


def rebuild_dir_path(path):
    """
    Rebuild path for current operating system.
    :param path: (string) path separated with / or \\
    :return: (string) rebuilt path
    """
    dirs = re.split(r"[\\/]", path)
    return os.path.join(*dirs)


def htk_time_to_ms(htk_time):
    """
    Convert HTK time to milliseconds.
    :param htk_time: (str) time in HTK format
    :return: (float) time in milliseconds
    """
    return float(htk_time) / 10000


# initialize parser
parser = argparse.ArgumentParser()

# add input and output paths arguments
parser.add_argument("--input_dir_path", help="input directory path")
parser.add_argument("--output_dir_path", help="output directory path")
parser.add_argument("--alignment_file", help="name of file with alignments")
parser.add_argument("--delimiter", help="delimiter between beginning time, end time and phoneme in alignment file")

# parse arguments
args = parser.parse_args()

# rebuild path for current operating system
input_dir_path = rebuild_dir_path(args.input_dir_path)
output_file_path = os.path.join(rebuild_dir_path(args.output_dir_path), os.path.basename(input_dir_path) + ".csv")

# estimate transition and self loop probabilities
phonemes = {}

with open(output_file_path, "w") as fw:
    with open(os.path.join(input_dir_path, args.alignment_file), 'r') as fr:
        for line in fr:
            line = line.strip()

            if line.startswith(("#!MLF!#", ".")) or line == "" or ".lab" in line:
                continue

            beginning_htk, end_htk, phoneme = line.split(args.delimiter.decode("string_escape"))[0:3]

            beginning_ms = htk_time_to_ms(beginning_htk)
            end_ms = htk_time_to_ms(end_htk)

            if phoneme not in phonemes:
                phonemes[phoneme] = []

            phonemes[phoneme].append(end_ms - beginning_ms)

    for phoneme, lengths in phonemes.iteritems():
        denominator = 1. * np.median(np.asarray(sorted(lengths))) / 10
        transition_proba = 1 / denominator
        self_loop_proba = 1 - transition_proba

        fw.write(",".join([str(col) for col in [phoneme, transition_proba, self_loop_proba]]))
        fw.write("\n")
