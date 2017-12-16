#!/bin/bash
module add python27-modules-intel
module add cuda

cd /storage/plzen1/home/mmajer/phoneme_classification

chmod +x ./run_model.sh
KERAS_BACKEND=theano THEANO_FLAGS=device=gpu,floatX=float32 ./run_model.sh
chmod -x ./run_model.sh
