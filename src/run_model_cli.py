import argparse
from collections import namedtuple
from lib.feeders import MLPFeeder, RNNFeeder

from lib.models import TestModelMLP
from lib.models import TestModelRNN

Model = namedtuple('Model', "model is_rnn")

MODELS = {
    "TestModelMLP": Model(model=TestModelMLP, is_rnn=False),
    "TestModelRNN": Model(model=TestModelRNN, is_rnn=True)
}

parser = argparse.ArgumentParser()

# add feeder arguments
parser.add_argument("--features_path", help="name of file containing features")
parser.add_argument("--ratio", help="ratio between train and val+test and between test and val, e.g. 0.75 0.5", nargs=2, type=float)
parser.add_argument("--test_speakers_path", help="name of text file with predefined test speakers, one per line (optional)", default=None)
parser.add_argument("--left_context", help="number of previous frames to build feature vectors", default=0, type=int)
parser.add_argument("--right_context", help="number of future frames to build feature vectors", default=0, type=int)
parser.add_argument("--time_steps", help="number of time steps in phoneme time series, applicable for RNN only", default=5, type=int)

# add model arguments
parser.add_argument("--model", help="model architecture")
parser.add_argument("--load", help="load saved model weights", action="store_true", default=False)
parser.add_argument("--epochs", help="number of epochs", type=int)
parser.add_argument("--batch_size", help="batch size", type=int)
parser.add_argument("--callbacks", help="model callbacks to use", nargs="+", default=[])

# parse arguments
args = parser.parse_args()

# load test speakers
test_speakers = []

if args.test_speakers_path:
    with open(args.test_speakers_path, "r") as fr:
        test_speakers = [speaker.strip() for speaker in fr.readlines()]

# choose model and corresponding feeder
print "\nBuilding temporary dataset..."

selected_model = MODELS[args.model]

if selected_model.is_rnn:
    feeder = RNNFeeder(args.features_path)

    feeder.remove_tmp_storage()
    feeder.create_datasets(tuple(args.ratio), args.time_steps, test_speakers=test_speakers, left_context=args.left_context, right_context=args.right_context)
else:
    feeder = MLPFeeder(args.features_path)

    feeder.remove_tmp_storage()
    feeder.create_datasets(tuple(args.ratio), test_speakers=test_speakers, left_context=args.left_context, right_context=args.right_context)

# train/load weights and predict
print "\nCompiling model..."

model = selected_model.model(feeder, args.epochs, args.batch_size, callbacks=args.callbacks)

print model.model.summary()

if args.load:
    print "\nLoading weights..."
    model.load_weights()
else:
    print "\nTraining..."
    model.train()

print "\nPredicting..."
predictions, transcriptions = model.predict()

# decode predictions and evaluate
# TODO: create Viterbi decoder

# clean after training and prediction
print "\nCleaning temporary dataset..."
feeder.remove_tmp_storage()




