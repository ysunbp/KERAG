# KERAG

This is the repository for the KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering.

## Install

```console
$ git clone [link to repo]
$ cd KERAG
$ pip install -r requirements.txt 
```

If you are using Anaconda, you can create a virtual environment and install all the packages:

```console
$ conda create --name KERAG python=3.9
$ conda activate KERAG
$ pip install -r requirements.txt
```

## Dataset

Set up the mock api server for CRAG, according to the instructions here: https://github.com/ysunbp/KERAG/tree/main/scripts/crag-mock-api

Download dataset and other folders from https://drive.google.com/drive/folders/1wHteCQSVQ3MI0fBFWrM84NOCiFafrZhH?usp=sharing

## Experiments

### CRAG
Please follow the these steps to train a summarizer and perform KBQA on CRAG
```console
$ cd scripts
$ python planner.py
$ python generate_summarizer_training_data.py
$ python summarizer_get_finetune_data.py
$ accelerate launch summarizer_finetune_fsdp.py
$ python CRAG-QA.py
```

### Head2Tail
```console
$ cd scripts
$ python -m torch.distributed.run --nproc_per_node 1 H2T-QA.py
```

### QALD-10-en
```console
$ cd scripts
$ python -m torch.distributed.run --nproc_per_node 1 QALD-QA.py
```
