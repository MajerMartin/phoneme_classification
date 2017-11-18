#!/bin/sh
# mark the shell script executable by running "chmod +x build_datasets.sh"
# run with "./build_datasets.sh"
cd src

printf "SkodaAuto - log-filterbank energies without deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SkodaAuto' \
--output_dir_path '../data/features' \
--wav_dir_name '_data8kHz-cut' \
--alignment_file 'SA_all_aligned.txt.phones.txt' \
--words_file 'SA_words.txt' \
--regex 'SkodaAuto' \
--encoding 'cp1250' \
--delimiter '\t' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--filters_count 40 \
lfe

printf "\nSkodaAuto - log-filterbank energies with deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SkodaAuto' \
--output_dir_path '../data/features' \
--wav_dir_name '_data8kHz-cut' \
--alignment_file 'SA_all_aligned.txt.phones.txt' \
--words_file 'SA_words.txt' \
--regex 'SkodaAuto' \
--encoding 'cp1250' \
--delimiter '\t' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--deltas \
--filters_count 40 \
lfe

printf "\nSkodaAuto - MFCC without deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SkodaAuto' \
--output_dir_path '../data/features' \
--wav_dir_name '_data8kHz-cut' \
--alignment_file 'SA_all_aligned.txt.phones.txt' \
--words_file 'SA_words.txt' \
--regex 'SkodaAuto' \
--encoding 'cp1250' \
--delimiter '\t' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--filters_count 26 \
mfcc \
--coeffs_count 13

printf "\nSkodaAuto - MFCC with deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SkodaAuto' \
--output_dir_path '../data/features' \
--wav_dir_name '_data8kHz-cut' \
--alignment_file 'SA_all_aligned.txt.phones.txt' \
--words_file 'SA_words.txt' \
--regex 'SkodaAuto' \
--encoding 'cp1250' \
--delimiter '\t' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--deltas \
--filters_count 26 \
mfcc \
--coeffs_count 13

printf "\nSpeechDat-E - log-filterbank energies without deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SpeechDat-E' \
--output_dir_path '../data/features' \
--wav_dir_name 'wavs' \
--alignment_file 'SD-E_all_phones_ALL_times.txt' \
--words_file 'SD-E_all_words.txt' \
--regex 'SpeechDat-E' \
--encoding 'cp1250' \
--delimiter ' ' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--filters_count 40 \
lfe

printf "\nSpeechDat-E - log-filterbank energies with deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SpeechDat-E' \
--output_dir_path '../data/features' \
--wav_dir_name 'wavs' \
--alignment_file 'SD-E_all_phones_ALL_times.txt' \
--words_file 'SD-E_all_words.txt' \
--regex 'SpeechDat-E' \
--encoding 'cp1250' \
--delimiter ' ' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--deltas \
--filters_count 40 \
lfe

printf "\nSpeechDat-E - MFCC without deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SpeechDat-E' \
--output_dir_path '../data/features' \
--wav_dir_name 'wavs' \
--alignment_file 'SD-E_all_phones_ALL_times.txt' \
--words_file 'SD-E_all_words.txt' \
--regex 'SpeechDat-E' \
--encoding 'cp1250' \
--delimiter ' ' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--filters_count 26 \
mfcc \
--coeffs_count 13

printf "\nSpeechDat-E - MFCC with deltas\n"
python build_dataset_cli.py \
--input_dir_path '../data/raw/SpeechDat-E' \
--output_dir_path '../data/features' \
--wav_dir_name 'wavs' \
--alignment_file 'SD-E_all_phones_ALL_times.txt' \
--words_file 'SD-E_all_words.txt' \
--regex 'SpeechDat-E' \
--encoding 'cp1250' \
--delimiter ' ' \
--frame_length 0.025 \
--frame_step 0.01 \
--normalize \
--deltas \
--filters_count 26 \
mfcc \
--coeffs_count 13