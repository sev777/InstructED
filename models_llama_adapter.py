import json

import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, LlamaConfig, AutoModelForCausalLM,GPT2Config,GPTJForCausalLM, AutoTokenizer,GPTJConfig,GPT2Tokenizer

from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int = 10
    adapter_layer: int = 30
    layer_st: int = 30
    layer_ed: int = 30


def Llama7B_adapter(args, **kwargs):


    model_path = args.llama_model_path


    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side='left'


    configuration = LlamaConfig.from_json_file(model_path+'/config.json')
    configuration.layer_st = args.adapter_st
    configuration.layer_ed = args.adapter_ed
    configuration.adapter_len = args.adapter_len
    configuration.args = args
    configuration.adapter = True if  args.use_adapter==1 else False

    model = LlamaForCausalLM.from_pretrained(
        model_path, config=configuration,device_map='auto'
    )


    # model_llama_adapter
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()

    for name, param in model.model.layers[args.adapter_st:args.adapter_ed].named_parameters():
        if ("gate" in name or "adapter" in name or 'ex' in name ) and 'gate_proj' not in name:
            param.data = param.data.float()
            param.requires_grad = True
        else:
            param.requires_grad = False
    print('\nTraining parameters')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    return model,tokenizer

def gpt2_adapter(args, **kwargs):


    model_path = args.llama_model_path

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id



    configuration = GPT2Config.from_json_file(model_path+'/config.json')
    configuration.layer_st = args.adapter_st
    configuration.layer_ed = args.adapter_ed
    configuration.adapter_len = args.adapter_len
    configuration.args = args
    configuration.adapter = True if  args.use_adapter==1 else False

    model = AutoModelForCausalLM.from_pretrained(model_path,config=configuration).cuda()


    # model_llama_adapter
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()

    for name, param in model.transformer.h[args.adapter_st:args.adapter_ed].named_parameters():
        if ("gate" in name or "adapter" in name or 'ex' in name ) and 'gate_proj' not in name:
            param.data = param.data.float()
            param.requires_grad = True
        else:
            param.requires_grad = False
    print('\nTraining parameters')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    return model,tokenizer


# set recommended archs
Llama7B_adapter = Llama7B_adapter
gpt2_adapter=gpt2_adapter



