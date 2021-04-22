#!/bin/sh
echo "This will unzip the tar dataset files"

tar -xvf /home/abrol/ASR/TensorFlowASR/scripts/Datasets/dev-clean.tar.gz -C /home/abrol/ASR/TensorFlowASR/scripts/Datasets/dev-clean/
tar -xvf /home/abrol/ASR/TensorFlowASR/scripts/Datasets/dev-other.tar.gz -C /home/abrol/ASR/TensorFlowASR/scripts/Datasets/dev-other

tar -xvf /home/abrol/ASR/TensorFlowASR/scripts/Datasets/test-clean.tar.gz -C /home/abrol/ASR/TensorFlowASR/scripts/Datasets/test-clean
tar -xvf /home/abrol/ASR/TensorFlowASR/scripts/Datasets/test-other.tar.gz -C /home/abrol/ASR/TensorFlowASR/scripts/Datasets/test-other

tar -xvf /home/abrol/ASR/TensorFlowASR/scripts/Datasets/train-clean-100.tar.gz -C /home/abrol/ASR/TensorFlowASR/scripts/Datasets/train-clean-100
tar -xvf /home/abrol/ASR/TensorFlowASR/scripts/Datasets/train-clean-360.tar.gz -C /home/abrol/ASR/TensorFlowASR/scripts/Datasets/train-clean-360
tar -xvf /home/abrol/ASR/TensorFlowASR/scripts/Datasets/train-clean-500.tar.gz -C /home/abrol/ASR/TensorFlowASR/scripts/Datasets/train-clean-500

