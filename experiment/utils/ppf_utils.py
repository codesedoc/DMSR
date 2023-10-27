from dataclasses import field

from nlpx.approach.transformer import TransformerModelArgument
from nlpx.argument import argument_class, ArgumentPool


Name2Backbone = {
    "bart": "facebook/bart-base",
    "t5": "t5-small"
}


@argument_class
class PGSTModelArgument(TransformerModelArgument):
    backbone_name: str = field(
        default= "t5",
        metadata={"help": "Specify the name of backbone"}
    )
    # def __post_init__(self):
    #     super().__post_init__()
    #     name, backbone = self.plm_name_or_path.split("-")
    #     assert name == ArgumentPool.meta_arguments.approach
    #     assert backbone in Name2Backbone
    #     self.backbone_name = backbone

    @property
    def backbone(self):
        return Name2Backbone[self.backbone_name]