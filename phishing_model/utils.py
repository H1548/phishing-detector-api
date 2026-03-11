from .paths import TOKENIZER_PATH
from sre_parse import Tokenizer 
import torch
import torch.nn as nn
from tokenizers import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenize = Tokenizer.from_file(TOKENIZER_PATH)


def tokenize_data(data, bos_token, eos_token, tokenize):
    for i in range(data.shape[0]):
        data[i] = [bos_token] + tokenize.encode(data[i]).ids + [eos_token]
    return data

def pad_sequence(sequence, max_length, pad_token_id):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_token_id] * (max_length - len(sequence))

def create_mask(x, pad_token):
    mask = x != pad_token
    mask = mask.to(torch.bool)
    return mask
