#!/bin/sh
# mark the shell script executable by running "chmod +x estimate_hmm_probas.sh"
# run with "./estimate_hmm_probas.sh"
cd src

OUTPUT_PATH='../data/hmm'

printf "SkodaAuto\n"
python estimate_hmm_probas_cli.py \
--input_dir_path '../data/raw/SkodaAuto' \
--output_dir_path ${OUTPUT_PATH} \
--alignment_file 'SA_all_aligned.txt.phones.txt' \
--delimiter '\t'

printf "SpeechDat-E\n"
python estimate_hmm_probas_cli.py \
--input_dir_path '../data/raw/SpeechDat-E' \
--output_dir_path ${OUTPUT_PATH} \
--alignment_file 'SD-E_all_phones_ALL_times.txt' \
--delimiter ' '