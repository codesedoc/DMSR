import datasets
import torch.distributions
from tqdm.auto import tqdm
from transformers import pipeline

from enum import Enum


class Sentiment(Enum):
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'
    NULL = 'NULL'

    def __str__(self):
        return str(self.value).upper()


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


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ModelBasedMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        self.classifier = pipeline("sentiment-analysis")

    def _compute(self, *, predictions=None, references=None, **kwargs):
        """Returns the scores"""
        result = {}
        result.update(self.process("pred", predictions))
        result.update(self.process("ref", references))

        return result

    def process(self, prefix, sentences):
        pole = []
        score = []
        total = len(sentences)
        for s in tqdm(sentences, total=total, desc="sentiment-analysis", disable=False):
            predicts = self.classifier(s)
            pole.append(Sentiment[predicts[0]['label']].value)
            score.append(predicts[0]['score'])

        # Skip a warning by set this flag to False
        torch.distributions.Distribution._validate_args = False

        return {
            f"{prefix}_pole": pole,
            f"{prefix}_score": score,
        }
