import argparse
from lib.preprocessors import Framer
from lib.extractors import LogFilterbankEnergies, MFCC
from lib.loaders import WaveLoader

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='feature_type')

# create subparsers for features
parser_lfe = subparsers.add_parser("lfe", help="log filterbank energies")
parser_mfcc = subparsers.add_parser("mfcc", help="mel-frequency cepstral coefficients")

# add input and output paths arguments
parser.add_argument("--input_dir_path", help="input directory path")
parser.add_argument("--output_dir_path", help="output directory path")
parser.add_argument("--wav_dir_name", help="name of directory containing wave files")
parser.add_argument("--alignment_file", help="name of file with alignments")
parser.add_argument("--words_file", help="name of file with words")

# add additional file information arguments
parser.add_argument("--regex", help="regular expression for matching speakers and utterances")
parser.add_argument("--encoding", help="words file encoding")
parser.add_argument("--delimiter", help="delimiter between beginning time, end time and phoneme in alignment file")

# add framer specific arguments
parser.add_argument("--frame_length", help="length of single frame in seconds", type=float)
parser.add_argument("--frame_step", help="length of frame step in seconds", type=float)

# add argument for normalization and delta coefficients
parser.add_argument("--normalize", help="normalize features column-wise", action="store_true", default=False)
parser.add_argument("--deltas", help="compute delta and delta-delta coefficients", action="store_true", default=False)

# add log filterbank energies and MFCC arguments
parser.add_argument("--filters_count", help="number of filters", type=int, default=26)

# add MFCC specific arguments
parser_mfcc.add_argument("--coeffs_count", help="number of coefficients to keep", type=int, default=13)

# parse arguments
args = parser.parse_args()

# initialize objects
framer = Framer(args.frame_length, args.frame_step)

if args.feature_type == "lfe":
    extractor = LogFilterbankEnergies(normalize=args.normalize, deltas=args.deltas, filters_count=args.filters_count)
elif args.feature_type == "mfcc":
    extractor = MFCC(normalize=args.normalize, deltas=args.deltas, filters_count=args.filters_count,
                     coeffs_count=args.coeffs_count)

loader = WaveLoader(framer, extractor)

# create dataset
loader.create_dataset(args.input_dir_path, args.output_dir_path, args.wav_dir_name, args.alignment_file,
                      args.words_file, args.regex, args.encoding, args.delimiter.encode().decode("unicode_escape"))
