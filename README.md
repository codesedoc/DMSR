# Learning Disentangled Meaning and Style Representations for Positive Text Reframing [:link:](https://aclanthology.org/2023.inlg-main.31/)

The code of is developed on [nlpe](https://github.com/codesedoc/nlpe), a python package for processing NLP experiments.

## Data
<ol>
    <li> MSCOCO: used to create the pseudo paraphrase generation dateset for PTR. </li>
    <li> Yelp: used to create the pseudo sentiment transfer dateset for PTR. </li>
    <li> PPF: used to evaluate methods for PTR. </li>
</ol>

Source of data: [MSCOCO](https://cocodataset.org/#home) ([version from here](https://github.com/IBM/quality-controlled-paraphrase-generation/tree/main/data/mscoco)), [Yelp](https://www.yelp.com/dataset) ([version from here](https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp)), [PPF](https://github.com/SALT-NLP/positive-frames)

## Enviroment Setup
### Requirements
<ul>
    <li> Git </li>
    <li> Python >= 3.11 </li>
    <li> CUDA 12.6 (if NVIDIA GPU is available) </li>
</ul>

### Step 1: Get Code and datasets
Clone the repository.
```shell
git clone https://github.com/codesedoc/DMSR.git
cd DMSR
```
### Step 2: Install Requirements
#### :large_blue_diamond: Conda
```shell
conda env create -f conda/environment_linux-64.yml # for linxu-64 platform
# conda env create -f conda/environment_oxs-arm64.yml # for oxs-arm64 (Mac Silicon) platform
# conda env create -f conda/environment_win-64.yml # for win-64 platform
conda activate dmsr
```
#### :large_blue_diamond: Python Virtual Environment
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Conduct Experiments

### Command to lauch the experiment for train, evaluation, and test.
```shell
python main.py -t ptr -d ppf --dataset_raw_dir storage/dataset/ppf/raw  -a dmsr --output_dir tmp --do_train --do_eval --do_predict 
```

### Explanation of Command Arguments:
#### Meta Argument of nlpe
<ul>
    <li> -d [name of dataset] </li>
    <li> -t [name of task] </li>
    <li> --dataset_raw_dir [path of dataset dir] </li>
    <li> -a [name of approach] </li>
    <li> --debug : Swith on debug mode</li>
</ul>

#### DMSR Argument
<ul>
    <li> --variant [name of variant of backbone model]: One of ("base", "st", "pg", "st2pg", "pg2st") is available, default is base.  </li>
    <li> --checkpoint [path of trained model]: It is used for only do evaluation on validation/test set. </li>
    <li> --backbone [name of PLM used as backbone model]: One of ("t5", "bart") is available, default is t5. </li>
</ul>

#### TrainerArgument of transformer
<ul>
    <li> --output_dir [path of output dir] </li>
    <li> --do_train: Conduct train</li>
    <li> --do_eval: Conduct evaluation on validation set</li>
    <li> --do_predict: Conduct  evaluation on test set</li>
    <!-- <li> --include_inputs_for_metrics=True: Deliver inputs as one argument during calculate metrics. </li> -->
</ul>

The other usages of transformers TrainiArgument can be referred to [here](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/trainer#transformers.TrainingArguments).


## BibTeX

```
@inproceedings{
    sheng-etal-2023-learning,
    title = "Learning Disentangled Meaning and Style Representations for Positive Text Reframing",
    author = "Sheng, Xu  and
      Fukumoto, Fumiyo  and
      Li, Jiyi  and
      Kentaro, Go  and
      Suzuki, Yoshimi",
    booktitle = "Proceedings of the 16th International Natural Language Generation Conference",
    month = sep,
    year = "2023",
    address = "Prague, Czechia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.inlg-main.31",
    pages = "424--430",
}
```