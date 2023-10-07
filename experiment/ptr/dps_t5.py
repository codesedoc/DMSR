from transformers import T5Config, T5Tokenizer, T5TokenizerFast, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration

from .dps_base import DPSConfig, DPSTokenizer, DPSTokenizerFast, DPSEncoder, DPSModelForPTR


class T5DPSConfig(DPSConfig, T5Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = False

    model_type = 'dps-t5'
    pass


class T5DPSTokenizer(DPSTokenizer, T5Tokenizer):
    pass



class T5DPSTokenizerFast(DPSTokenizerFast, T5TokenizerFast):
    slow_tokenizer_class = T5DPSTokenizer

    def preprocess_input_sequence(self, sequence):
        if isinstance(sequence, list):
            return [self.preprocess_input_sequence(s) for s in sequence]
        elif isinstance(sequence, str):
            prefix = "summarize: "
            return prefix + sequence
        else:
            raise ValueError


class T5DPSEncoder(DPSEncoder, T5Stack):
    @property
    def backbone_layers(self):
        return self.block

    @backbone_layers.setter
    def backbone_layers(self, value):
        assert isinstance(type(self.block), value)
        self.block = value


class T5DPSModelForPTR(DPSModelForPTR, T5ForConditionalGeneration):
    @property
    def backbone_encoder(self):
        return self.encoder

    config_class = T5DPSConfig

    def __init__(self, config: DPSConfig):
        super().__init__(config)
        self.encoder = T5DPSEncoder(config, self.shared)


AutoConfig.register(T5DPSConfig.model_type, T5DPSConfig)
AutoTokenizer.register(T5DPSConfig, T5DPSTokenizer, T5DPSTokenizerFast)
AutoModelForSeq2SeqLM.register(T5DPSConfig, T5DPSModelForPTR)
