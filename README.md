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
    <li> Python (>=3.11) </li>
</ul>

### Step 1: Get Code and datasets
Clone the repository.
```shell
git clone git@github.com:codesedoc/DMSR.git
cd DMSR
```
### Step 2: Install Requirements
#### :large_blue_diamond: Conda
```shell
conda create -n dmsr python=3.11
conda activate dmsr
```
#### :large_blue_diamond: Python Virtual Environment
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
To protect system data during running docker container, it is recommended to creat a user belong to docker group, but without root permission.
Running follow command can create an account name "docker-1024"!

` bash sh/docker-1024 `

Running follow command to build the image of basic environment of NLPx. 

` docker compose build nlpx-env`

To use Nvidia GPU in docker containers, please install the "NVIDIA Container Toolkit" referring to [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt).

#### Environment Variables

There are some necessary variables used during building images. They are defined in the file ".env" 

## Conduct Experiments
Running follow command to start 

` bash sh/docker-run `

To conduct different variants of method in paper, please define the value of variable "ARGS_FILE" to the path of experiment argument file.

## Citation

```
@inproceedings{sheng-etal-2023-learning,
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
    abstract = "The positive text reframing (PTR) task which generates a text giving a positive perspective with preserving the sense of the input text, has attracted considerable attention as one of the NLP applications. Due to the significant representation capability of the pre-trained language model (PLM), a beneficial baseline can be easily obtained by just fine-tuning the PLM. However, how to interpret a diversity of contexts to give a positive perspective is still an open problem. Especially, it is more serious when the size of the training data is limited. In this paper, we present a PTR framework, that learns representations where the meaning and style of text are structurally disentangled. The method utilizes pseudo-positive reframing datasets which are generated with two augmentation strategies. A simple but effective multi-task learning-based model is learned to fuse the generation capabilities from these datasets. Experimental results on Positive Psychology Frames (PPF) dataset, show that our approach outperforms the baselines, BART by five and T5 by six evaluation metrics. Our source codes and data are available online.",
}
```