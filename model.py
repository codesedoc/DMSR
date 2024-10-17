import os.path
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import Optional, Any, Dict, Union, Tuple, Callable, Set
from functools import partial
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
                        #  BartTokenizer, BartTokenizerFast
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.modeling_t5 import T5Stack, T5Config, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder, BartConfig, BartForConditionalGeneration

from nlpe import ArgumentPool
from nlpe.utils import Glossary
from dataclasses import dataclass, field

from utils import GlossaryEnum, get_unified_model_type_str


class BackboneName(StrEnum):
    BART = auto()
    T5 = auto()
    

BackboneName2HFPath = {
    BackboneName.BART: "facebook/bart-base",
    BackboneName.T5: "t5-small"
}

    
class DMSRModel(PreTrainedModel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.dynamic_layers, nn.ModuleList)
        
        self.backbone_arg: BackboneArgument = ArgumentPool()["backbone_argument"]

        if self.backbone_arg.variant != VariantGlossaryEnum.BASE.value and self.backbone_arg.checkpoint_origin == CheckpointOrigin.LOCAL:        
            self.pg_layers = deepcopy(self.dynamic_layers)
            self.st_layers = deepcopy(self.dynamic_layers)
        else:
            self.pg_layers = self.st_layers = None
            
        self._dynamict_layers_block_types = [type(i) for i in self.dynamic_layers]
    
    def init_dynamic_layers_from_pretraind(self):
        if self.pg_layers is None:
            self.pg_layers: nn.Module = deepcopy(self.dynamic_layers)
        if self.st_layers is None:
            self.st_layers: nn.Module = deepcopy(self.dynamic_layers)
        
        assert isinstance(self.dynamic_layers, nn.ModuleList) and \
            isinstance(self.pg_layers, type(self.dynamic_layers)) and \
                isinstance(self.st_layers, type(self.dynamic_layers))
        
        def _check_two_modules(ma, mb):
            assert isinstance(ma, nn.Module) and isinstance(mb, nn.Module)
            ma_m_list = list(ma.modules())
            mb_m_list = list(mb.modules())
            assert len(ma_m_list) == len(mb_m_list)
            for ma_m, mb_m in zip(ma_m_list, mb_m_list):
                assert id(ma_m) != id(mb_m)
            
            ma_p_list = list(ma.parameters(True))
            mb_p_list = list(mb.parameters(True))  
            assert len(ma_p_list) == len(mb_p_list)
            for ma_p, mb_p in zip(ma_p_list, mb_p_list):
                assert id(ma_p) != id(mb_p)
                
        _check_two_modules(self.pg_layers, self.dynamic_layers)
        _check_two_modules(self.dynamic_layers, self.st_layers)
        _check_two_modules(self.pg_layers, self.st_layers)
    
    @property
    def dynamic_layers(self):
        raise RuntimeError()
    
    @abstractmethod
    def _set_dynamic_layers(self, layers):
        raise RuntimeError()

    def set_dynamic_layers(self, layers: nn.ModuleList):
        if layers is not None:
            assert isinstance(layers, nn.ModuleList) and len(layers) == len(self._dynamict_layers_block_types)
            for l, t in zip(layers, self._dynamict_layers_block_types):
                assert isinstance(l, t)
                # breakpoint()
        self._set_dynamic_layers(layers)
    
    @staticmethod
    def _clone_layers(layers: torch.nn.ModuleList):
        assert isinstance(layers, torch.nn.ModuleList)
        layer_class = type(layers[0])
        assert issubclass(layer_class, torch.nn.Module)
        result = torch.nn.ModuleList(layer_class() for l in layers)
        result.load_state_dict(layers.state_dict())
        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *args, **kwargs):
        transformer: DMSRModel = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if transformer.backbone_arg.variant != VariantGlossaryEnum.BASE.value:
            transformer.init_dynamic_layers_from_pretraind()
        return transformer


class T5DMSRConfig(T5Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = False
        
    model_type = get_unified_model_type_str(BackboneName.T5)
    pass


class T5DMSRModel(DMSRModel, T5ForConditionalGeneration):
    config_class = T5DMSRConfig
    def _set_dynamic_layers(self, layers):
        self.encoder.block = layers
        
    @property
    def dynamic_layers(self):
        return self.encoder.block
    

class BartDMSRConfig(BartConfig):
    
    model_type = get_unified_model_type_str(BackboneName.BART)
    pass


class BartDMSRModel(DMSRModel, BartForConditionalGeneration):
    
    config_class = BartDMSRConfig
    
    def _set_dynamic_layers(self, layers):
        self.model.encoder.layers = layers
    
    @property
    def dynamic_layers(self):
        return self.model.encoder.layers

    
def _regist_classes(config_class, tokenizer_class, tokenizer_fast_class, model_class):
    assert issubclass(config_class, PretrainedConfig)
    assert issubclass(tokenizer_class, PreTrainedTokenizer)
    assert issubclass(tokenizer_fast_class, PreTrainedTokenizerFast)
    assert issubclass(model_class, PreTrainedModel)
    AutoConfig.register(config_class.model_type, config_class)
    AutoTokenizer.register(config_class, tokenizer_class, tokenizer_fast_class)
    AutoModelForSeq2SeqLM.register(config_class, model_class)


AutoModelForSeq2SeqLM.from_pretrained
_regist_classes(T5DMSRConfig, T5Tokenizer, T5TokenizerFast, T5DMSRModel)
_regist_classes(BartDMSRConfig, BartTokenizer, BartTokenizerFast, BartDMSRModel)


_ID2DynamicMergedLayers = dict()


def _merge_two_sides(first_layers: nn.ModuleList, second_layers: nn.ModuleList) -> tuple:
    _id = str(id(first_layers)) + str(id(second_layers))
    if _id not in _ID2DynamicMergedLayers:
        assert isinstance(first_layers, nn.ModuleList)
        assert isinstance(second_layers, nn.ModuleList)
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
        _ID2DynamicMergedLayers[_id] = nn.ModuleList(first2second_layers), nn.ModuleList(second2first_layers)
    return _ID2DynamicMergedLayers[_id]


class VariantGlossaryEnum(GlossaryEnum):
    PG = auto()
    ST =  auto()
    ST2PG =  auto()
    PG2ST =  auto()
    INCORPORATE =  auto()
    BASE =  auto()
    
    
def dynamic_layers_in_forward(model: DMSRModel, variant_glossary:Glossary) -> nn.ModuleList:
    assert isinstance(variant_glossary, Glossary)
    match(variant_glossary):
        case VariantGlossaryEnum.PG.value:
            return model.pg_layers
        case VariantGlossaryEnum.ST.value:
            return model.st_layers
        case VariantGlossaryEnum.ST2PG.value:
            return _merge_two_sides(model.st_layers, model.pg_layers)[0]
        case VariantGlossaryEnum.PG2ST.value:
            return _merge_two_sides(model.pg_layers, model.st_layers)[0]
        case VariantGlossaryEnum.INCORPORATE.value:
            raise ValueError("Do not support INCORPORATE without overwrite the 'farword' method of DMSRModel or its Encoder!")
            # return partial(_double_side_forward, model, model.st_layers, model.pg_layers)
        case VariantGlossaryEnum.BASE.value:
            return model.dynamic_layers
    raise ValueError(f"Value of variant ({variant_glossary.value}) is invalid")


class CheckpointOrigin(StrEnum):
    LOCAL = auto()
    HF = auto()
    
    
@dataclass
class BackboneArgument:
    backbone: BackboneName = field(
        default= BackboneName.T5,
        metadata={
            "help": "Specify the name of backbone",
            "choices" : [b.value for b in BackboneName]
        }
    )
    
    checkpoint: str = field(
        default= None,
        metadata={
            "help": "Specify the dir of checkpont",
        }
    )
    
    variant: VariantGlossaryEnum = field(
        default= VariantGlossaryEnum.BASE.value,
        metadata={
            "help": "Specify the variant of backbone",
            "choices" : [str(v.value) for v in VariantGlossaryEnum]
        }
    )
    
    @property
    def name(self):
        return self.backbone
    
    def __post_init__(self, *args, **kwargs):
        trainer_args = ArgumentPool()["trainer_argument"]
        if isinstance(self.variant, str):
            self.variant = VariantGlossaryEnum(str).value
        if self.checkpoint is None:
            self.checkpoint = trainer_args.output_dir
        try: 
            if trainer_args.do_train:
                raise ValueError("Always training from the HF checkpont.")
            self.checkpoint = Path(self.checkpoint)
            assert self.checkpoint.is_dir()
            self.checkpoint_origin = CheckpointOrigin.LOCAL
        except:
            self.checkpoint = BackboneName2HFPath[self.name]
            self.checkpoint_origin = CheckpointOrigin.HF

    @property
    def model_type_str(self) -> str:
        return get_unified_model_type_str(self.name)