#!/bin/sh
echo "This will create santiy transcripts for you, Kindly change the datapaths accordingly"

INPUT_DIR_1="Datasets/train-clean-100/LibriSpeech/train-clean-100"
OUTPUT_FILE="Datasets/train-clean-100-sanity/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE -s

INPUT_DIR_1="Datasets/dev-clean/LibriSpeech/dev-clean"
OUTPUT_FILE="Datasets/dev-clean-sanity/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE -s

INPUT_DIR_1="Datasets/dev-other/LibriSpeech/dev-other"
OUTPUT_FILE="Datasets/dev-other-sanity/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE -s

INPUT_DIR_1="Datasets/test-clean/LibriSpeech/test-clean"
OUTPUT_FILE="Datasets/test-clean-sanity/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE -s
