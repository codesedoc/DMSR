from transformers import BartConfig, BartTokenizer, BartTokenizerFast, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.bart.modeling_bart import BartEncoder, BartForConditionalGeneration

from .dps_base import DPSConfig, DPSTokenizer, DPSTokenizerFast, DPSEncoder, DPSModelForPTR


class BartDPSConfig(DPSConfig, BartConfig):
    model_type = 'dps-bart'
    pass


class BartDPSTokenizer(DPSTokenizer, BartTokenizer):
    pass


class BartDPSTokenizerFast(DPSTokenizerFast, BartTokenizerFast):
    slow_tokenizer_class = BartDPSTokenizer


class BartDPSEncoder(DPSEncoder, BartEncoder):
    @property
    def backbone_layers(self):
        return self.layers

    @backbone_layers.setter
    def backbone_layers(self, value):
        assert isinstance(type(self.layers), value)
        self.layers = value


class BartDPSModelForPTR(DPSModelForPTR, BartForConditionalGeneration):
    @property
    def backbone_encoder(self):
        return self.model.encoder

    config_class = BartDPSConfig

    def __init__(self, config: DPSConfig):
        super().__init__(config)
        self.model.encoder = BartDPSEncoder(config, self.model.shared)


AutoConfig.register(BartDPSConfig.model_type, BartDPSConfig)
AutoTokenizer.register(BartDPSConfig, BartDPSTokenizer, BartDPSTokenizerFast)
AutoModelForSeq2SeqLM.register(BartDPSConfig, BartDPSModelForPTR)
