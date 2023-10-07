#!/bin/bash

source $(dirname $0)/init.sh

if [ -z $ROOT_DIR ]; then echo "ROOT_DIR is empty" && exit 1; fi
cd $ROOT_DIR
# bash sh/download.sh -n ppf
echo 'Welcome to NLPx'

export TOKENIZERS_PARALLELISM=true
ARGS_FILE=$1
if [ -z $ARGS_FILE ]; then ARGS_FILE="experiment/test/args-test.json"; fi

echo "run main.py with the args_file($ARGS_FILE)"
python main.py --args_file "$ARGS_FILE"

echo 'NLPx runing finished!'
