#!/bin/bash
module add python34-modules-gcc
module add cuda-8.0
module load cudnn-6.0

export PYTHONPATH="/storage/plzen1/home/mmajer/.local/lib/python3.4/site-packages/:$PYTHONPATH"

cd /storage/plzen1/home/mmajer/phoneme_classification

mkdir -p logs/jobs
DATE=`date '+%Y-%m-%d_%H:%M:%S'`

chmod +x ./run_model.sh
./run_model.sh > logs/jobs/${DATE}.txt
chmod -x ./run_model.sh




