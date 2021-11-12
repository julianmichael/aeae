from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("text_classifier_with_distribution")
class TextClassifierWithDistributionPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a class distribution for it.  In particular, it can be used with
    the [`BasicClassifier`](../models/basic_classifier.md) model.

    Originally copied from https://github.com/allenai/allennlp/blob/main/allennlp/predictors/text_classifier.py.

    Registered as a `Predictor` with name "text_classifier".
    """

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model and adds the `"label"` (maximum probability
        class) and `"label_distribution"` (full distribution) to the output.
        """
        sentence = json_dict["sentence"]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        # TODO @margsli test this
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        label_probs = outputs["probs"]
        new_instance.add_field("model_label", LabelField(int(label), skip_indexing=True))
        new_instance.add_field("model_label_probs", LabelField(label_probs, skip_indexing=True))
        return [new_instance]
