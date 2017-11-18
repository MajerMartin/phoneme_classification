#!/bin/sh
# mark the shell script executable by running "chmod +x run_model.sh"
# run with "./run_model.sh"
cd src

python run_model_cli.py \
--features_path ../data/features/SkodaAuto_25_10_log_filterbank_energies_deltas.hdf5 \
--ratio 0.75 0.5 \
--left_context 2 \
--right_context 1 \
--time_steps 5 \
--model TestModelRNN \
--epochs 3 \
--batch_size 32 \
--callbacks tensorboard modelCheckpoint earlyStopping #reduceLROnPlateau # \
#--test_speakers_path test_speakers.txt
#--load
