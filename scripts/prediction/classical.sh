#!/bin/bash
#SBATCH --job-name=classical-eval
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=bdata
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=8:00:00

# I use source to initialize conda into the right environment.
source ~/.bashrc
cd /gscratch/zlab/margsli/gitfiles/aeae

cat $0
echo "--------------------"

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/chaosnli_mnli_m.jsonl save/classical1/model.tar.gz data/build/chaosnli/mnli_m/dev.jsonl

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/chaosnli_snli.jsonl save/classical1/model.tar.gz data/build/chaosnli/snli/dev.jsonl

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/snli.jsonl save/classical1/model.tar.gz data/build/snli/dev.jsonl

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/mnli.jsonl save/classical1/model.tar.gz data/build/mnli/dev.jsonl

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/fever_nli.jsonl save/classical1/model.tar.gz data/build/fever_nli/dev.jsonl

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/anli_r1.jsonl save/classical1/model.tar.gz data/build/anli/r1/dev.jsonl

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/anli_r2.jsonl save/classical1/model.tar.gz data/build/anli/r2/dev.jsonl

allennlp predict --include-package aeae --predictor nli_classifier --cuda-device -1 --batch-size 1 \
--output-file save/classical1/preds/anli_r3.jsonl save/classical1/model.tar.gz data/build/anli/r3/dev.jsonl