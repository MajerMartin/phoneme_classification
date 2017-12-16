#!/bin/bash
module add python27-modules-intel
module add cuda

cd /storage/plzen1/home/mmajer/phoneme_classification

chmod +x ./run_model.sh
./run_model.sh
chmod -x ./run_model.sh
