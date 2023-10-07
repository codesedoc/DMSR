from typing import Dict, Any

import datasets
import nltk
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


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PPFMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        assert nltk.download('punkt')
        self.bleu = load_metric("sacrebleu", experiment_id=self.experiment_id)
        self.rouge = load_metric("rouge", experiment_id=self.experiment_id)
        self.bscore = load_metric("bertscore", experiment_id=self.experiment_id)
        self.perplexity = load_metric("perplexity", experiment_id=self.experiment_id)

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

        # assert nltk.download('punkt')
        # self.bleu = load_metric("sacrebleu", experiment_id=self.experiment_id)
        # self.rouge = load_metric("rouge", experiment_id=self.experiment_id)
        # self.bscore = load_metric("bertscore", experiment_id=self.experiment_id)
        # self.perplexity = load_metric("perplexity", experiment_id=self.experiment_id)
        pass

        from textblob.sentiments import NaiveBayesAnalyzer
        # self.naive_bayes_analyzer = NaiveBayesAnalyzer()

    def _compute(self, *, predictions=None, references=None, inputs:  np.ndarray = None, **kwargs) -> Dict[str, Any]:
        # Rouge expects a newline after each sentence
        assert len(predictions) == len(references)
        predictions_joined = []
        references_joined = []
        for pred, label in zip(predictions, references):
            if len(pred.strip()) == 0:
                continue
            predictions_joined.append("\n".join(nltk.sent_tokenize(pred.strip())))
            references_joined.append("\n".join(nltk.sent_tokenize(label.strip())))

        result = self.rouge.compute(predictions=predictions_joined,
                                     references=references_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        references_expanded = [[x] for x in references]
        result2 = self.bleu.compute(predictions=predictions, references=references_expanded)
        result['sacrebleu'] = round(result2["score"], 1)

        bertscores = self.bscore.compute(predictions=predictions, references=references, lang="en", batch_size=kwargs.get("batch_size", 32))
        result['bertscore'] = np.array(bertscores["f1"]).mean()

        perplexity = self.perplexity.compute(model_id='gpt2', add_start_token=False, input_texts=predictions)
        result['perplexity'] = perplexity["mean_perplexity"]

        deltaTB = []
        from textblob import TextBlob
        for pred, inp in zip(predictions, inputs):
            deltaTB.append(
                # TextBlob(pred, analyzer=self.naive_bayes_analyzer).sentiment.p_pos - \
                # TextBlob(ref, analyzer=self.naive_bayes_analyzer).sentiment.p_pos
                TextBlob(pred).sentiment.polarity - TextBlob(inp).sentiment.polarity
            )
        result['deltaTB'] = np.array(deltaTB).mean()

        # print(result2)


        return {k: round(v, 4) for k, v in result.items()}
