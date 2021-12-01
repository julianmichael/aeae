# An Ambiguous Evaluation of Adversarial Evaluation

![AEAEAEAEAE](assets/aeae.png)
<div align="center"><em>AEAEAEAEAE</em></div>

## Contents

* `aeae/`: Source code for data, metrics, etc.
* `proposal/`: Completed project proposal.

## Usage

This project requires Python 3 and is written using AllenNLP and PyTorch.

**Workstation setup:**
* Start with `python scripts/download.py` from the base directory to download
datasets.
* Install Python dependencies with `pip install -r requirements.txt`.
* Preprocess datasets with `python scripts/build_data.py`.

To sanity-check model training, run
```
MODE=tiny allennlp train config/basic.jsonnet --include-package aeae -o '{"trainer.cuda_device": -1}' -s save/tiny
```
This will train a model on a tiny subset of MNLI using CPU. Changing MODE accordingly uses
different data sources (see [basic.jsonnet](config/basic.jsonnet)) the cuda device determines 
which GPU is used.

## Documentation

NLI instances are preprocessed into the following format:
```
{
  "uid": String,
  "premise": String,
  "hypothesis": String,
  "label": "e" | "c" | "n"
}
```
TODO: specify format for the label distribution.


