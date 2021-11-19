from typing import List, Dict
import torch

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("nli_classifier")
class NliClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in a premise and hypothesis and returns
    a class distribution for it.  In particular, it can be used with
    the [`NliClassifier`](../models/nli_classifier.py) model.

    Originally copied from https://github.com/allenai/allennlp/blob/main/allennlp/predictors/text_classifier.py.

    Registered as a `Predictor` with name "nli_classifier".
    """

    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        return self.predict_json({"premise": premise, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"premise": "...", "hypothesis": "..."}`.
        Runs the underlying model.
        """
        premise = json_dict["premise"]
        hypothesis = json_dict["hypothesis"]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            premise = tokenizer.tokenize(premise)
            hypothesis = tokenizer.tokenize(hypothesis)
        return self._dataset_reader.text_to_instance(premise, hypothesis)
