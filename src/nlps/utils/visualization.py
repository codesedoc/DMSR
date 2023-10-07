import hashlib
import os
from enum import Enum
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.nlps.utils.utils import set_seed


def visualize_scatter(index: np.ndarray, color_list, name=None, output_dir:str=None, show:bool=False, f_size=(6.4, 6.4), **kwargs):
    plt.figure(figsize=f_size)
    plt.subplot()
    if name is None or not isinstance(name, str):
        name = "visualization"
    plt.scatter(index[:, 0], index[:, 1], c=color_list, label=name, s=kwargs.get("dot_size", 20))
    plt.legend()
    path = os.path.join(output_dir, f"{name}_tsne.png")

    path_dir = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(path_dir):
        os.system(f"mkdir -p {path_dir}")
    plt.savefig(path, dpi=120)

    if show:
        plt.show()


def visualize_text(index: np.ndarray, text: np.ndarray, color_list, name=None, output_dir:str=None, show:bool=False, f_size=(6.4, 6.4), **kwargs):
    plt.figure(figsize=f_size)
    plt.subplot()
    if name is None or not isinstance(name, str):
        name = "visualization"
    x_i = np.indices((len(index), 1))
    y_i = np.indices((len(index), 1))
    y_i[1] = y_i[1]+1
    plt.xlim(max(-PLOT_SIZE, int(np.floor(np.min(index[x_i])))-2), min(PLOT_SIZE, int(np.ceil(np.max(index[x_i])))+2))
    plt.ylim(max(-PLOT_SIZE, int(np.floor(np.min(index[y_i])))-2), min(PLOT_SIZE, int(np.ceil(np.max(index[y_i])))+2))
    for axis, t, c in zip(index, text, color_list):
        plt.text(axis[0], axis[1], t, c=c, label=name)
    plt.title(name)
    # plt.legend()
    path = os.path.join(output_dir, f"{name}_tsne.png")
    path_dir = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(path_dir):
        os.system(f"mkdir -p {path_dir}")
    plt.savefig(path, dpi=120)
    if show:
        plt.show()


class VisualizationMode(Enum):
    SCATTER = 'scatter'
    TEXT = 'text'
    CURVE = 'curve'


PLOT_SIZE = 160


def visualize_embeddings(embeddings: np.ndarray, color_list, name=None, mode=VisualizationMode.SCATTER, output_dir:str=None, show:bool=False, f_size=(6.4, 6.4), text_list=None, seed=1, **kwargs):
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.double)
    assert len(embeddings.shape) == 2
    default_perplexity = 30

    set_seed(seed)
    tsne = TSNE(n_components=2, perplexity=min(default_perplexity, len(embeddings) - 1))
    if isinstance(output_dir, str) and len(output_dir) > 0:
        if not os.path.isdir(output_dir):
            os.system(f"mkdir -p '{output_dir}'")

    result = tsne.fit_transform(embeddings)

    if name is None or not isinstance(name, str):
        name = "embedding"
    else:
        name += "-embedding"

    if mode is VisualizationMode.SCATTER:
        visualize_scatter(result, color_list, name, output_dir, show, f_size, **kwargs)

    elif mode is VisualizationMode.TEXT:
        if text_list is None:
            raise ValueError
        visualize_text(result, text_list, color_list, name, output_dir, show, f_size, **kwargs)
    else:
        raise ValueError


class Embedding:
    def __init__(self, _uuid, embedding):
        self.uuid = _uuid
        self.embedding = embedding
        self.order = None
        self.inset_order = None
    def export(self, mode="json"):
        if mode == "json":
            result = {
                "embedding": self.embedding,
                "uuid": self.uuid,
                "dimension": len(self.embedding),
                "order": self.order,
                "inset_order": self.inset_order
            }
        else:
            raise ValueError

        return result


class SentenceEmbedding(Embedding):
    def __init__(self,  sentence, embedding, _uuid=None):

        _uuid = _uuid if _uuid is not None else hashlib.md5(str(sentence).encode("utf-8")).hexdigest()
        super(SentenceEmbedding, self).__init__(_uuid, embedding)
        self.sentence = sentence

    def export(self, mode="json"):
        if mode == "json":
            result = super(SentenceEmbedding, self).export(mode)
            result.update(
                {
                    "sentence": self.sentence
                }
            )
        else:
            raise ValueError

        return result


class EmbeddingTable:
    def __init__(self, space, task, approach, name="null"):
        self.space = space
        self.name = name
        self.approach = approach
        self.task = task
        self._embedding_container: dict = {}

    @property
    def head(self):
        return {
            "id": f"{self.approach}_{self.task}_{self.space}_{self.size}",
            "name": self.name,
            "task": self.task,
            "approach": self.approach,
            "space": self.space,
            "size": self.size
        }

    @property
    def size(self):
        return len(self.embeddings)

    @property
    def embeddings(self):
        result = []
        for i, k in enumerate(sorted(self._embedding_container)):
            embedding = self._embedding_container[k]
            embedding.order = i
            result.append(embedding)
        return result

    def add_embedding(self, embedding: Embedding):
        result = 0
        united_id = embedding.uuid
        repeated_times = 20
        while united_id in self._embedding_container:
            result = 1
            repeated_times -= 1
            if repeated_times == 0:
                raise ValueError
            united_id = hashlib.md5(str(united_id).encode("utf-8")).hexdigest()
        embedding.inset_order = len(self._embedding_container)
        self._embedding_container[united_id] = embedding
        return result

    def pop_embedding(self, uuid):
        return self._embedding_container.pop(uuid, None)

    def get_embedding(self, uuid):
        return self._embedding_container.get(uuid, None)

    def export(self, mode="json"):
        if mode == "json":
            result = {
                "head": self.head,
                "body": [
                    e.export(mode)
                    for e in self.embeddings
                ]
            }
        else:
            raise ValueError
        return result