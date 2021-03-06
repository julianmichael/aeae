#!/bin/bash
#SBATCH --job-name=classical-train
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# I use source to initialize conda into the right environment.
source ~/.bashrc
cd /gscratch/zlab/margsli/gitfiles/aeae

cat $0
echo "--------------------"

MODE=classical allennlp train config/basic.jsonnet --include-package aeae -s save/classical1 -o '{"trainer.cuda_device": 0}'