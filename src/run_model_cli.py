import os
import argparse
from collections import namedtuple
from lib.feeders import MLPFeeder, RNNFeeder
from lib.decoders import LanguageModel, Decoder

from lib.models import PyramidDropoutMLP
from lib.models import DropoutLSTM
from lib.models import DropoutGRU

Model = namedtuple('Model', "model is_rnn")

MODELS = {
    "PyramidDropoutMLP": Model(model=PyramidDropoutMLP, is_rnn=False),
    "DropoutLSTM": Model(model=DropoutLSTM, is_rnn=True),
    "DropoutGRU": Model(model=DropoutGRU, is_rnn=True)
}

parser = argparse.ArgumentParser()

# add feeder arguments
parser.add_argument("--features_path", help="name of file containing features")
parser.add_argument("--ratio", help="ratio between train and val+test and between test and val, e.g. 0.75 0.5", nargs=2,
                    type=float)
parser.add_argument("--test_speakers_path",
                    help="name of text file with predefined test speakers, one per line (optional)", default=None)
parser.add_argument("--left_context", help="number of previous frames to build feature vectors", default=0, type=int)
parser.add_argument("--right_context", help="number of future frames to build feature vectors", default=0, type=int)
parser.add_argument("--time_steps", help="number of time steps in phoneme time series, applicable for RNN only",
                    default=5, type=int)
parser.add_argument("--sample", help="train, validation and test size limits", nargs=3, default=[], type=int)

# add model arguments
parser.add_argument("--model", help="model architecture")
parser.add_argument("--load", help="load saved model weights", action="store_true", default=False)
parser.add_argument("--epochs", help="number of epochs", type=int)
parser.add_argument("--batch_size", help="batch size", type=int)
parser.add_argument("--callbacks", help="model callbacks to use", nargs="+", default=[])

# add language model arguments
parser.add_argument("--ngram", help="ngram order", default=0, type=int)

# parse arguments
args = parser.parse_args()

# load test speakers
test_speakers = []

if args.test_speakers_path:
    with open(args.test_speakers_path, "r") as fr:
        test_speakers = [speaker.strip() for speaker in fr.readlines()]

# choose model and corresponding feeder
print("\nBuilding temporary dataset...")

selected_model = MODELS[args.model]

if selected_model.is_rnn:
    feeder = RNNFeeder(args.features_path)

    feeder.remove_tmp_storage()
    feeder.create_datasets(tuple(args.ratio), args.time_steps, test_speakers=test_speakers,
                           left_context=args.left_context, right_context=args.right_context, sample=args.sample)
else:
    feeder = MLPFeeder(args.features_path)

    feeder.remove_tmp_storage()
    feeder.create_datasets(tuple(args.ratio), test_speakers=test_speakers, left_context=args.left_context,
                           right_context=args.right_context, sample=args.sample)

# train/load weights and predict
print("\nCompiling model...")

model = selected_model.model(feeder, args.epochs, args.batch_size, callbacks=args.callbacks)

print(model.model.summary())

if args.load:
    print("\nLoading weights...")
    model.load_weights()
else:
    print("\nTraining...")
    model.train()

print("\nEvaluating...")
print(model.evaluate())

print("\nPredicting...")
predictions, transcriptions = model.predict()

# create directory for results
print("\n\nCreating directory for results...")

features_name = os.path.splitext(os.path.basename(feeder.features_path))[0]
model_name = model.__class__.__name__

results_dir = os.path.join("..", "results", features_name, model_name)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# build language model
print("\nBuilding language model...")

languageModel = LanguageModel(feeder)
language_model = languageModel.create_model(ngram=args.ngram)

is_bigram = args.ngram == 2

# decode and save results
print("\nDecoding predictions...")

pred_path = os.path.join(results_dir, "pred.mlf")
ref_path = os.path.join(results_dir, "ref.mlf")

decoder = Decoder(feeder, language_model, is_bigram)

with open(pred_path, "w") as fw_pred, open(ref_path, "w") as fw_ref:
    fw_pred.write("#!MLF!#\n")
    fw_ref.write("#!MLF!#\n")

    for i, (observations, transcription) in enumerate(zip(predictions, transcriptions)):
        print("\r  {0}/{1}".format(i, len(predictions) - 1), end=" ")

        decoded_transcription = decoder.decode(observations)

        fw_pred.write('"*/{}.rec"\n'.format(i))
        fw_ref.write('"*/{}.lab"\n'.format(i))

        fw_pred.write("\n".join(decoded_transcription))
        fw_ref.write("\n".join(transcription))

        fw_pred.write("\n.\n")
        fw_ref.write("\n.\n")

with open(os.path.join(results_dir, "phonemes"), "w") as fw:
    fw.write("\n".join(feeder.encoder.classes_))

# clean after training and prediction
print("\n\nCleaning temporary dataset...")
feeder.remove_tmp_storage()
