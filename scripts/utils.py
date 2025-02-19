#!/usr/bin/env python

import os
from transformers import LlamaTokenizerFast, AutoTokenizer

# crag competition provided tokenizer
#tokenizer_path =  "/export/data/LLM-benchmark-project-KB/LLMs/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
TOKENIZER = LlamaTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 75
    tokenized_prediction = TOKENIZER.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
    trimmed_prediction = TOKENIZER.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def get_token_length(prediction):
    tokenized_prediction = TOKENIZER.encode(prediction)
    return len(tokenized_prediction) - 1


def trim_predictions_to_token_length(prediction, max_token_length):
    tokenized_prediction = TOKENIZER.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
    trimmed_prediction = TOKENIZER.decode(trimmed_tokenized_prediction)
    return trimmed_prediction