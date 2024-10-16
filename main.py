from datetime import datetime
import json
from pathlib import Path
import os
from utils import jsonable, log_segment, set_random_seed
import sys
from nlpe import ArgumentPool, Data, EvaluatorProxy, ArgumentFactory
import logging


os.environ["HF_HOME"] = str(Path(ArgumentPool().meta_argument["cache_dir"], "huggingface"))


from nlpe.utils import Glossary, global_logger
from transformers import Seq2SeqTrainingArguments, HfArgumentParser
from dataclasses import asdict
from dmrs import DMSR, BackboneArgument
from dataset import PPF_EVALUATOR, DatasetGlossaryEnum, DatasetInfo, merge_datasets


if sys.version_info.major != 3 or sys.version_info.minor < 11:
    print(f"The current python version is: \n {sys.version} \nMinimum requirement: 3.11 !")
    raise RuntimeError("Python Version Error!")


def parse_argument():
    ArgumentPool().push(ArgumentFactory(
        argument_glossary="trainer_argument",
        argument_type= Seq2SeqTrainingArguments,
        process=lambda : HfArgumentParser(Seq2SeqTrainingArguments).parse_args_into_dataclasses(return_remaining_strings=True)[0], 
        to_dict=asdict)
    )
    
    if ArgumentPool().meta_argument["debug"]:
        ArgumentPool()["trainer_argument"].output_dir = Path(ArgumentPool()["trainer_argument"].output_dir, time_suffix)
    ArgumentPool()["trainer_argument"].run_name = time_suffix
    ArgumentPool().push(ArgumentFactory(
        argument_glossary="backbone_argument",
        argument_type= BackboneArgument,
        process=lambda : HfArgumentParser(BackboneArgument).parse_args_into_dataclasses(return_remaining_strings=True)[0], 
        to_dict=asdict)
    )
    log_segment(logger.info, title="All Arguments", content=json.dumps(jsonable(ArgumentPool().all_args), indent=4))

                
def run():
    seed = ArgumentPool()['trainer_argument'].seed
    logger.info(f"Set Random Seed {seed}")
    set_random_seed(seed)
    logger.info(f"Initionlize Approach {ArgumentPool().meta_argument['approach']}")
    approach = DMSR()
    logger.info(f"Load and Merge Pseudo Datasets {[str(g) for g in DatasetGlossaryEnum.pseudo_glossaries()]}")
    merged_dataset = merge_datasets([DatasetInfo(g) for g in DatasetGlossaryEnum.pseudo_glossaries()])
    pseudo_data = merged_dataset.to_data()
    
    logger.info("Process Stage 1")
    approach.process(pseudo_data, stage=1)
    
    dataset_glossary = None
    available_glossaries = set(p.value for p in DatasetGlossaryEnum) - set(p for p in DatasetGlossaryEnum.pseudo_glossaries())
    for g in available_glossaries:
        if ArgumentPool().meta_argument["dataset"] == str(g):
            dataset_glossary = g
            break
                
    if not isinstance(dataset_glossary, Glossary):
        logger.warning(f" Dataset name ({ArgumentPool().meta_argument['dataset']}) is not available, it should be one in {[str(g) for g in available_glossaries]}")
        raise ValueError
        
    logger.info(f"Load Dataset {dataset_glossary}")
    dataset_info = DatasetInfo(dataset_glossary)
    data = dataset_info.to_data()
    
    logger.info("Process Stage 2")
    data.all_datasets
    approach.process(data, stage=2)


if __name__ == '__main__':
    time_suffix = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logger = global_logger()
    if ArgumentPool().meta_argument["debug"]:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    parse_argument()
    # breakpoint()
    try:
        run()
    except:
        if ArgumentPool().meta_argument["debug"]:
            ouput_dir = Path(ArgumentPool()['trainer_argument'].output_dir)
            ouput_dir.rmdir()
        raise


## import required packages explictly to help the requirements.txt generate tools detecting them

import sentencepiece