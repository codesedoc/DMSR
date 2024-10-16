from pathlib import Path
from typing import Any, Dict
import evaluate
import numpy as np
import datasets
import nltk
from nlpe.utils import global_logger
from nlpe import ArgumentPool

_DESCRIPTION=""
_KWARGS_DESCRIPTION=""
@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PPFMetric(evaluate.Metric):
    """TODO: Short description of my metric."""

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        
        self.bleu = evaluate.load("sacrebleu", experiment_id=self.experiment_id, seed=kwargs.get("seed", None))
        self.rouge = evaluate.load("rouge", experiment_id=self.experiment_id, seed=42)
        self.bscore = evaluate.load("bertscore", experiment_id=self.experiment_id, seed=kwargs.get("seed", None))
        self.perplexity = evaluate.load("perplexity", experiment_id=self.experiment_id, seed=kwargs.get("seed", None))

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return evaluate.MetricInfo(
            module_type="metric",
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            description="",
            citation="",
        )

    def _download_and_prepare(self, dl_manager):
        nltk_data_path = str(Path(ArgumentPool().meta_argument["cache_dir"], "nltk_data"))
        nltk.download('punkt_tab', download_dir=nltk_data_path)
        nltk.data.path.append(nltk_data_path)
        # assert nltk.download('punkt')
        # self.bleu = load_metric("sacrebleu", experiment_id=self.experiment_id)
        # self.rouge = load_metric("rouge", experiment_id=self.experiment_id)
        # self.bscore = load_metric("bertscore", experiment_id=self.experiment_id)
        # self.perplexity = load_metric("perplexity", experiment_id=self.experiment_id)
        pass

        # from textblob.sentiments import NaiveBayesAnalyzer
        # self.naive_bayes_analyzer = NaiveBayesAnalyzer()

    def _compute(self, *, predictions=None, references=None, inputs:  np.ndarray = None, **kwargs) -> Dict[str, Any]:
        # Rouge expects a newline after each sentence
        assert len(predictions) == len(references)
        logger = global_logger()
        predictions_joined = []
        references_joined = []

        for pred, label in zip(predictions, references):
            if len(pred.strip()) == 0:
                continue
            predictions_joined.append("\n".join(nltk.sent_tokenize(pred.strip())))
            references_joined.append("\n".join(nltk.sent_tokenize(label.strip())))

        logger.info("********** Calculate Rouge, SacreBLEU, BERTScore, Perplexities **********")
        result = self.rouge.compute(predictions=predictions_joined,
                                     references=references_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        references_expanded = [[x] for x in references]
        
        result2 = self.bleu.compute(predictions=predictions, references=references_expanded)
        result['sacrebleu'] = round(result2["score"], 1)

        bertscores = self.bscore.compute(predictions=predictions, references=references, lang="en", batch_size=kwargs.get("batch_size", 32))
        result['bertscore'] = np.array(bertscores["f1"]).mean()

        perplexity = self.perplexity.compute(model_id='gpt2', add_start_token=False, predictions=predictions)
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
