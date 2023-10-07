from abc import abstractmethod, ABC
from collections import OrderedDict
from enum import Enum
from sklearn import metrics
from typing import Dict, Any

import datasets
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
from math import ceil

def to_batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


class ClassificationType(Enum):
    BINARY = 'binary'
    MCSL = 'mcsl'
    MCML = 'mcml'


class Evaluator(ABC):
    @abstractmethod
    def compute(self, predictions: np.ndarray = None, references: np.ndarray = None, **kwargs):
        raise NotImplementedError


class BinaryEvaluator(Evaluator):

    def compute(self, predictions: np.ndarray = None, references: np.ndarray = None, **kwargs):
        raise ValueError


class MCSLEvaluator(Evaluator):
    def compute(self, predictions: np.ndarray = None, references: np.ndarray = None, **kwargs):
        raise ValueError


class MCMLEvaluator(Evaluator):
    def compute(self, predictions: np.ndarray = None, references: np.ndarray = None, **kwargs):
        result = OrderedDict([
            ('micro_f1', metrics.f1_score(predictions, references, average='micro', zero_division=0)),
            ('macro_f1', metrics.f1_score(predictions, references, average='macro', zero_division=0)),
        ])
        return result


TASK_TYPE2EVALUATOR_CLASS = {
    ClassificationType.MCML.value: MCMLEvaluator
}

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ClassificationMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._classification_type = None

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference

            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value('int32')),
                'references': datasets.Sequence(datasets.Value('int32')),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    @property
    def classification_type(self) -> ClassificationType:
        if not hasattr(self, '_classification_type'):
            raise ValueError('Please set the attribute at first!')
        return self._classification_type

    @classification_type.setter
    def classification_type(self, value: ClassificationType):
        # assert isinstance(value, ClassificationType)
        assert ClassificationType(value.value)
        self._classification_type = value

    @property
    def evaluator(self) -> Evaluator:
        result = TASK_TYPE2EVALUATOR_CLASS[self.classification_type.value]()
        return result

    def _compute(self, *, predictions: np.ndarray = None, references: np.ndarray = None, inputs: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        # Rouge expects a newline after each sentence

        predictions = np.array(predictions)
        references = np.array(references)

        assert predictions.shape == references.shape
        evaluator = self.evaluator
        assert isinstance(evaluator, Evaluator)
        result = evaluator.compute(predictions, references)

        return {k: round(v, 4) for k, v in result.items()}
