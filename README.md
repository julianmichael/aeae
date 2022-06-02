# Overconfidence in the Face of Ambiguity with Adversarial Data

![AEAEAEAEAE](assets/aeae.png)
<div align="center"><em>AEAEAEAEAE</em></div>

## Contents

This is the repository for the paper:

**Overconfidence in the Face of Ambiguity with Adversarial Data.**
Margaret Li\* and Julian Michael,\*
_Proceedings of the First Workshop on Dynamic Adversarial Data Collection (DADC)_ at NAACL 2022.

(The silly acronym is from the original working name, "An Ambiguous Evaluation of Adversarial Evaluation")

In this repository:
* `aeae/`: Source code for data, metrics, etc.
* `scripts/`: Entry points for running predictions, evaluating, and producing plots for our analysis.

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

Rest of the documentation is TODO.
