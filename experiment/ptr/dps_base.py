import dataclasses
import dataclasses
import inspect
import os.path
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Optional, Any, Dict, Union, Tuple, Callable, Set

import torch
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn
from transformers import PreTrainedModel, \
    PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.nlps.data import TextData


class Encoder_Path_Type(Enum):
    PG = "pg"
    ST = "st"
    ST2PG = "st2pg"
    PG2ST = "pg2st"
    CROSS = "cross"
    BASE = "base"

    @staticmethod
    def _single_side_forward(encoder: nn.Module, forward_layers: nn.ModuleList, *args, **kwargs) -> Any:
        encoder: DPSEncoder = encoder
        layers_tmp = encoder.backbone_layers
        encoder.backbone_layers = forward_layers
        forward_method = kwargs.pop("forward_method")
        output = forward_method(*args, **kwargs)
        encoder.backbone_layers = layers_tmp
        return output

    @staticmethod
    def _double_side_forward(encoder: nn.Module, first_layers: nn.ModuleList, second_layers: nn.ModuleList, *args,
                            **kwargs) -> Any:
        encoder: DPSEncoder = encoder
        first2second_layers, second2first_layers = Encoder_Path_Type._cross_double_sides(first_layers, second_layers)
        kwargs['output_hidden_states'] = True
        first2second_output = Encoder_Path_Type._single_side_forward(encoder, first2second_layers, *args, **kwargs)
        second2first_output = Encoder_Path_Type._single_side_forward(encoder, second2first_layers, *args, **kwargs)
        output = []
        for i in range(len(first2second_output)):
            f_o, s_o = first2second_output[i], second2first_output[i]
            if isinstance(f_o, torch.Tensor):
                pooling = encoder.cross_path_pooling((1, 2))
                output.append(torch.squeeze(pooling(torch.stack([f_o, s_o], -1)), -1))
            else:
                output.append((f_o, s_o))
        if type(first2second_output) is not type:
            output = type(first2second_output)(*output)
        else:
            type(first2second_output)(output)
        return output

    @staticmethod
    def _cross_double_sides(first_layers: nn.ModuleList, second_layers: nn.ModuleList) -> tuple:
        assert len(first_layers) == len(second_layers)
        length = len(first_layers)
        first2second_layers = []
        second2first_layers = []
        for i in range(length // 2):
            first2second_layers.extend([first_layers[2 * i], second_layers[2 * i + 1]])
            second2first_layers.extend([second_layers[2 * i], first_layers[2 * i + 1]])
        if len(first_layers) % 2:
            first2second_layers.append(first_layers[-1])
            second2first_layers.append(second_layers[-1])

        assert len(first2second_layers) == len(second2first_layers) and len(first2second_layers) == length

        return nn.ModuleList(first2second_layers), nn.ModuleList(second2first_layers)

    def forward_call_back(self, encoder: nn.Module, *args, **kwargs) -> Any:
        encoder: DPSEncoder = encoder
        if self is Encoder_Path_Type.PG:
            output = self._single_side_forward(encoder, encoder.pg_layers, *args, **kwargs)

        elif self is Encoder_Path_Type.ST:
            output = self._single_side_forward(encoder, encoder.st_layers, *args, **kwargs)

        elif self is Encoder_Path_Type.ST2PG:
            layer_tmp = self._cross_double_sides(encoder.st_layers, encoder.pg_layers)[0]
            output = self._single_side_forward(encoder, layer_tmp, *args, **kwargs)

        elif self is Encoder_Path_Type.PG2ST:
            layer_tmp = self._cross_double_sides(encoder.st_layers, encoder.pg_layers)[1]
            output = self._single_side_forward(encoder, layer_tmp, *args, **kwargs)

        elif self is Encoder_Path_Type.CROSS:
            output = self._double_side_forward(encoder, encoder.st_layers, encoder.pg_layers, *args, **kwargs)

        elif self is Encoder_Path_Type.BASE:
            output = self._single_side_forward(encoder, encoder.backbone_layers, *args, **kwargs)
        else:
            raise ValueError

        return output


@dataclasses.dataclass
class TaskCase:
    name: str
    prediction_to_output: Callable
    path_type: Encoder_Path_Type = None
    data: TextData = None
    is_main_task: bool = False

    def to_dict(self):
        return {
            "name": self.name,
            "path_type": self.path_type.value,
            "data": self.data.abbreviation,
        }


class DPSConfig(PretrainedConfig):
    def __init__(self, *args, **kwargs):
        self.dynamic_taskcase: TaskCase = None
        self._name2taskcase: Dict[str, TaskCase] = None
        super().__init__(*args, **kwargs)


    @property
    def task_names(self):
        if self.name2taskcase is None:
            return None

        return [n for n in self.name2taskcase.keys()]

    @property
    def name2taskcase(self):
        return self._name2taskcase

    @name2taskcase.setter
    def name2taskcase(self, value: Dict):
        assert value is None or isinstance(value, Dict)
        self._name2taskcase = value

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.pop("dynamic_taskcase", None)
        result.pop("_name2taskcase", None)
        return result


class DPSTokenizer(PreTrainedTokenizer):
    pass


class DPSTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = DPSTokenizer

    def preprocess_input_sequence(self, sequence):
        return sequence


class DPSModelForPTR(PreTrainedModel):
    config_class = DPSConfig

    @property
    @abstractmethod
    def backbone_encoder(self):
        pass

    @staticmethod
    def _deepcopy_layers(layers: torch.nn.ModuleList, config: PretrainedConfig):
        assert isinstance(layers, torch.nn.ModuleList)
        assert isinstance(config, PretrainedConfig)
        layer_class = type(layers[0])
        assert issubclass(layer_class, torch.nn.Module)
        depth = len(layers)
        duplicate_layers = deepcopy(layers)
        # duplicate_layers = nn.ModuleList([layer_class(config) for _ in range(depth)])
        # duplicate_layers.load_state_dict(layers.state_dict())
        return duplicate_layers

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert isinstance(kwargs.get('config', None), DPSConfig)
        transformer: DPSModelForPTR = super().from_pretrained(pretrained_model_name_or_path,
                                                              config=kwargs.get('config'))
        encoder = transformer.backbone_encoder
        config: DPSConfig = encoder.config
        encoder.pg_layers = cls._deepcopy_layers(encoder.backbone_layers, config)
        encoder.st_layers = cls._deepcopy_layers(encoder.backbone_layers, config)

        return transformer

    def forward(self, input_ids=None, attention_mask=None, labels=None, task=None, decoder_attention_mask=None,
                **kwargs):
        config: DPSConfig = self.config
        config.dynamic_taskcase = config.name2taskcase[task]
        output: Seq2SeqLMOutput = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                               decoder_attention_mask=decoder_attention_mask, **kwargs)
        return output

    def prepare_inputs_for_generation(self, *args, **model_kwargs):
        result = super().prepare_inputs_for_generation(*args, **model_kwargs)
        if "task" in model_kwargs:
            result["task"] = model_kwargs["task"]
        return result


class DPSEncoder(PreTrainedModel):
    def __init__(self, config: DPSConfig, embed_tokens: Optional[nn.Embedding] = None, *args, **kwargs):
        super().__init__(config, embed_tokens=embed_tokens, *args, **kwargs)
        self.pg_layers = None
        self.st_layers = None
        self.cross_path_pooling = torch.nn.MaxPool2d

    @property
    @abstractmethod
    def backbone_layers(self):
        pass

    @backbone_layers.setter
    @abstractmethod
    def backbone_layers(self, value):
        pass

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise ValueError

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        raise ValueError

    def _reorder_cache(self, past, beam_idx):
        raise ValueError

    def forward(self, *args, **kwargs) -> Any:
        from transformers.models.bart.modeling_bart import BartEncoder
        signature = inspect.signature(super().forward)
        kw = {n: kwargs.get(n, None) for n in signature.parameters}
        kw ["forward_method"] = super().forward
        path_type = self.config.dynamic_taskcase.path_type
        output = path_type.forward_call_back(self, *args, **kw)
        assert output is not None
        return output


class ContrastiveLoss:
    def __init__(self):
        self.batch_size = None
        self._loss = None

    @property
    def loss(self):
        if self._loss is None:
            self._loss = NTXentLoss()
        return self._loss

    def _create_contrastive_pos_location(self, batch_size):
        return torch.cat(
            [
                torch.cat([
                    torch.zeros(batch_size, batch_size).byte(),
                    torch.diag(torch.ones(batch_size)).byte()
                ], dim=1),
                torch.cat([
                    torch.diag(torch.ones(batch_size)).byte(),
                    torch.zeros(batch_size, batch_size).byte()
                ], dim=1)
            ],
        )

    def compute(self, seq_embeddings, seq_embeddings_shadow):
        loss_func = self.loss
        # breakpoint()
        batch_size = seq_embeddings.size(dim=0)
        p_location = self._create_contrastive_pos_location(batch_size)
        n_location = 1 - p_location
        n_location.fill_diagonal_(0)

        assert torch.sum(p_location).item() == batch_size * 2
        assert torch.sum(n_location).item() == (batch_size * 2) * ((batch_size * 2) - 2)

        p_location = p_location.to(device=seq_embeddings.device)
        n_location = n_location.to(device=seq_embeddings.device)
        embeddings = torch.cat([seq_embeddings, seq_embeddings_shadow])

        indices_tuple = list(torch.where(p_location))
        indices_tuple.extend(torch.where(n_location))
        indices_tuple = tuple(indices_tuple)
        # print(f"*******************{embeddings.shape}****************************")
        try:
            loss = loss_func(embeddings, indices_tuple=indices_tuple)
        except Exception:
            print(f"*******************{embeddings.shape}****************************")
            raise

        return loss

