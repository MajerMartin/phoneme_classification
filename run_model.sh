#!/bin/sh
# mark the shell script executable by running "chmod +x run_model.sh"
# run with "./run_model.sh"
cd src

python run_model_cli.py \
--features_path ../data/features/SkodaAuto_25_10_log_filterbank_energies.hdf5 \
--ratio 0.75 0.5 \
--left_context 0 \
--right_context 0 \
--time_steps 10 \
--model DropoutLSTM \
--epochs 100 \
--batch_size 32 \
--gpu \
--callbacks tensorboard modelCheckpoint reduceLROnPlateau CSVLogger # \
#--test_speakers_path test_speakers.txt
#--load
