import argparse
import os
import random

import numpy as np
import logging

from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, AutoTokenizer, AutoModel, GPT2Tokenizer,AutoModelForCausalLM,GPT2Config,LlamaTokenizerFast
import json

import time


import torch


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]



base_root = '/root/siton-data-9aa46a4f0e354f65bd8679947a35e67e/LMs/LMs/huggingface' # Base root for the original LLM
parser = argparse.ArgumentParser("MAE pre-training", add_help=False)

parser.add_argument("--adapter_path",
                    default="./output_dir/hop_withnogate_v2/checkpoint-8.pth",  # the reuslt path after training the model.
                    type=str, help="Name of model to train")
parser.add_argument("--adapter_type", default="no_gate", type=str, help="no_gate or FFN") # the layer type the instruct are.
parser.add_argument("--model_tp", default="/open_llama_7b_v2", type=str, help="Name of model to train") # the model type
parser.add_argument("--setting", default="", type=str, help="Name of model to train")



args = parser.parse_args()

model_path = base_root + args.model_tp
adapter_path = args.adapter_path
local_rank = 0
world_size = 1
temperature: float = 0.1
top_p: float = 0.75
max_seq_len: int = 512
max_batch_size: int = 32
quantizer: bool = False
PROMPT_DICT = {
    "ins_prompt": (

        "Instruction:\n{ins}\nInput:\n{questions[0]}"
    ),
    "only_ins_input_hop": (
        "Instruction:\n{ins}\nInput:\n{prompt_hop}"
    ),
    "only_ins_input": (
        "Instruction:\n{ins}\nInput:\n{rephrase_prompt}"
    ),
    "only_ins_input_loc": (
        "Instruction:\n{ins}\nInput:\n{locality_prompt}"
    ),
    "only_ins_input_ed": (
        "Instruction:\n{ins}\nInput:\n{prompt}"
    ),
}


ret_tok= AutoTokenizer.from_pretrained(base_root+'/contriever-msmarco') #load contriever-msmarco
ret_model = AutoModel.from_pretrained(base_root+'/contriever-msmarco').cuda()





def get_sent_embeddings_r(sents, contriever, tok, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def mean_pooling_r(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
# cos=torch.nn.CosineSimilarity()
def retrieve_facts_r(query, fact, contriever, tok, k=5,r=False):

    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact.T)[0]
    knn = sim.topk(k, largest=True)

    return knn


def make_mem_r(data):
    new_facts=[d['ins'] for d in data]

    embs = get_sent_embeddings(new_facts, ret_model, ret_tok)
    return embs, ret_model, ret_tok,new_facts

def get_sent_embeddings(sents, contriever, tok, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def make_mem(data):
    try:
        new_facts=[d['ins'] for d in data]
    except:

        new_facts = [rw['cloze']+' '+rw['answer'] for d in data for rw in d['new_single_hops']]

   
    embs = get_sent_embeddings(new_facts, ret_model, ret_tok)
    return embs, ret_model, ret_tok,new_facts

def load_models():

    model_args = torch.load(args.adapter_path, map_location="cpu")['args']
    if 'llama' in args.model_tp:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer.cls_token_id = tokenizer.eos_token_id
        tokenizer.cls_token = tokenizer.eos_token

        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.sep_token = tokenizer.eos_token

        tokenizer.mask_token_id = tokenizer.eos_token_id
        tokenizer.mask_token = tokenizer.eos_token

        configuration = LlamaConfig.from_json_file(model_path + '/config.json')
        configuration.layer_st = model_args.adapter_st
        configuration.layer_ed = model_args.adapter_ed
        configuration.adapter_len = model_args.adapter_len
        configuration.args = model_args
        configuration.adapter = True if model_args.use_adapter == 1 else False

        model = LlamaForCausalLM.from_pretrained(
            model_path, config=configuration,device_map='auto'
        )
       
        model.load_state_dict(torch.load(adapter_path,map_location={'cuda:1': 'cuda:0'})['model'], strict=False)
    else: #GPT2
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.cls_token_id = tokenizer.eos_token_id
        tokenizer.cls_token = tokenizer.eos_token

        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.sep_token = tokenizer.eos_token

        tokenizer.mask_token_id = tokenizer.eos_token_id
        tokenizer.mask_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        configuration = GPT2Config.from_json_file(model_path + '/config.json')
        configuration.layer_st = model_args.adapter_st
        configuration.layer_ed = model_args.adapter_ed
        configuration.adapter_len = model_args.adapter_len
        configuration.args = model_args
        configuration.adapter = True if model_args.use_adapter == 1 else False
        model = AutoModelForCausalLM.from_pretrained(model_path, config=configuration).cuda()
   
        model.load_state_dict(torch.load(adapter_path)['model'], strict=False)
    return model,tokenizer


def slice_list(matrix, start_indices, left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index - 1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0] - 1:-1]
        else:
            return matrix[start_indices[0]:]


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
cos=torch.nn.CosineSimilarity()
def retrieve_facts(query, fact, contriever, tok, k=5,r=False):
    if r:
        inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
        sim = (query_emb @ fact.T)[0]
        knn = sim.topk(k, largest=True)

        return knn
    else:
        inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
        fact_embs = tok([fact], padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
            outputs = contriever(**fact_embs)
            fact_embs = mean_pooling(outputs[0], fact_embs['attention_mask']).cpu()

        sim = (query_emb @ fact_embs.T)[0].tolist()[0]


        cos_sim=cos(query_emb, fact_embs).tolist()[0]

        return sim, cos_sim


def test_prediction_acc(model, tok, prompts, targets, device, locality=False,o_lc=None,head=''):

    if ret_model is not None:
        Ins,inp = prompts.split('\nInput:')
        ins_lens = [len(tok.encode(Ins))]
        sims = [retrieve_facts(Ins.replace('Instruction:\n',''), inp.replace('\n',''), ret_model, ret_tok)]
        ins_lens=[ins_lens,sims]
    else:
        ins_lens=True
    if isinstance(prompts, str):
        prompts, targets = [prompts, ], [targets, ]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts, targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1


    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok,ins_lens=ins_lens,output_attentions=True)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers, prompt_len, left=True)
        labels = slice_list(labels, prompt_len, left=False)
        answers_de = tok.decode(answers).lower()
        labels_de = tok.decode(labels).lower()

        if locality:
            lc_outputs = model(**prompt_target_tok,ins_lens=False)

            if type(lc_outputs) is torch.Tensor:
                lclogits = lc_outputs
            else:
                lclogits = lc_outputs.logits

            lc_ans= slice_list(torch.argmax(lclogits, dim=-1).squeeze().detach().cpu().numpy().tolist(), prompt_len, left=True)

            if isinstance(o_lc, str):
                o_lc = [o_lc, ]
            prompt_target = [prompt + ' ' + target for prompt, target in zip(o_lc, targets)]
            prompt_target_tok = tok(
                prompt_target,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(f"cuda:{device}")
            prompt_tok = tok(
                o_lc,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
            num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
            prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]

            olc_outputs = model(**prompt_target_tok,ins_lens=False)

            if type(olc_outputs) is torch.Tensor:
                olclogits = olc_outputs
            else:
                olclogits = olc_outputs.logits


            olc_ans= slice_list(torch.argmax(olclogits, dim=-1).squeeze().detach().cpu().numpy().tolist(), prompt_len, left=True)


            return np.mean(np.equal(answers, lc_ans)), np.mean(np.equal(olc_ans, answers)), np.mean(
            np.equal(answers, labels)), answers, lc_ans, olc_ans


        if isinstance(answers[0], list):
            res = []
            for ans, label in zip(answers, labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res, answers, labels
        else:
            return np.mean(np.equal(answers, labels)), tok.encode(answers_de), tok.encode(
                labels_de), answers_de, labels_de,answers, labels


def get_ann_res(model,tokenizer,LOG,ds=-1):
    ann = json.load(
        open('./data/counterfact_portability_gpt4_ins.json'))[:ds] #Counterfact path

    log_dic = {}

    res_dict = {'edit': [], 're': [], 'loc': [], 'hop': []}
    log_dic['edit'] = []
    log_dic['loc'] = []
    log_dic['re'] = []
    log_dic['hop'] = []


    hop_tk_score = []
    ed_tk_score = []
    re_tk_score = []
    loc_tk_score = []
    retri_acc={
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }
    all_res = {
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }
    embs, md, tk, nfs = make_mem(ann)
    for ai, a in enumerate(ann):

        try:
            a['rephrase_prompt'] = random.choice(a['paraphrase_prompts'])
            a['prompt'] = a['requested_rewrite']['prompt'].format(a['requested_rewrite']['subject'])
            a['locality_prompt'] = random.choice(a['neighborhood_prompts'])
            locality_ans = a['requested_rewrite']['target_true']['str']
            new_targe = a['requested_rewrite']['target_new']['str']
        except:
            a['rephrase_prompt'] = a['rephrase']
            a['prompt'] = a['src']
            a['locality_prompt'] = a['loc']
            locality_ans = a['loc_ans']
            new_targe = a['alt']
        a['oins'] = a['ins']
        # eval the edit
        fact_ids = retrieve_facts_r(a['prompt'], embs, md, tk, r=True)
        if torch.std(fact_ids[0]) > 0.1:
            a['ins'] = nfs[fact_ids[1][0]]
            edit_input = PROMPT_DICT["only_ins_input_ed"].format_map(a)
            ed_res_tk = test_prediction_acc(model, tokenizer, edit_input, new_targe, 0)
            ed_tk_score.append(ed_res_tk)
        else:
            # ed_tk_score.append([0])
            ed_res_tk = test_prediction_acc(model, tokenizer, edit_input, new_targe, 0, locality=True, o_lc=a['prompt'])
            re_tk_score.append([ed_res_tk[1]])

        # # eval the rephrase
        fact_ids = retrieve_facts(a['rephrase_prompt'], embs, md, tk, r=True)
        if torch.std(fact_ids[0]) > 0.1:
            a['ins'] = nfs[fact_ids[1][0]]
            re_input = PROMPT_DICT["only_ins_input"].format_map(a)
            re_res_tk = test_prediction_acc(model, tokenizer, re_input, new_targe, 0)
            re_tk_score.append(re_res_tk)
        else:
            # re_tk_score.append([0])
            re_res_tk = test_prediction_acc(model, tokenizer, re_input, new_targe, 0, locality=True,o_lc=a['rephrase_prompt'])
            re_tk_score.append([re_res_tk[1]])

        # eval the loc
        fact_ids = retrieve_facts(a['locality_prompt'], embs, md, tk, r=True)
        if torch.std(fact_ids[0]) > 0.1:
            a['ins'] = nfs[fact_ids[1][0]]
            loc_input = PROMPT_DICT["only_ins_input_loc"].format_map(a)
            loc_res_tk = test_prediction_acc(model, tokenizer, loc_input, locality_ans, 0, locality=True,
                                             o_lc=a['locality_prompt'])
            loc_tk_score.append(loc_res_tk)
        else:
            loc_tk_score.append([1, 1, 1])

        # eval the hop
        if 'portability' in a:
            a['prompt_hop'] = a['portability']['New Question']
            portability_ans = a['portability']['New Answer']
            fact_ids = retrieve_facts(a['prompt_hop'], embs, md, tk, r=True)
            if torch.std(fact_ids[0]) > 0.1:
                a['ins'] = nfs[fact_ids[1][0]]

                pro_input = PROMPT_DICT["only_ins_input_hop"].format_map(a)
                pro_res_token = test_prediction_acc(model, tokenizer, pro_input, portability_ans, 0)
                hop_tk_score.append(pro_res_token)
            else:
                pro_res_token=test_prediction_acc(model, tokenizer, pro_input, portability_ans, 0, locality=True,
                                    o_lc=a['prompt_hop'])
                hop_tk_score.append([pro_res_token[1]])


    LOG.info(f'********** Finlly ct {args.setting}---{len(ann)}  ********** ')
    # LOG.info(f'hop_score: {sum([e for e in hop_score]) / len(hop_score)}\n')
    LOG.info(f'hop_tk_score: {sum([e[0] for e in hop_tk_score]) / len(hop_tk_score)}\n')
    # LOG.info(f'edit_score: {sum([e for e in edit_score]) / len(edit_score)}\n')
    LOG.info(f'ed_tk_score: { sum([e[0] for e in ed_tk_score]) / len(ed_tk_score)}\n')
    # LOG.info(f're_score: {sum([e for e in re_score]) / len(re_score)}\n')
    LOG.info(f're_tk_score: {sum([e[0] for e in re_tk_score]) / len(re_tk_score)}\n')
    # LOG.info(f'loc_score: {sum([e for e in loc_score]) / len(loc_score)}\n')
    LOG.info(f'loc_tk_with_ins_score: {sum([e[0] for e in loc_tk_score]) / len(loc_tk_score)}\n')
    LOG.info(f'loc_with_pre_tk_score: {sum([e[1] for e in loc_tk_score]) / len(loc_tk_score)}\n')
    LOG.info(f'loc_with_pre_trues_score: {sum([e[2] for e in loc_tk_score]) / len(loc_tk_score)}\n')
    LOG.info(f'loc_with_re_and_trues: {sum([max(e[2],e[1],e[0]) for e in loc_tk_score]) / len(loc_tk_score)}\n')

    all_res['ed'].append(ed_tk_score)
    all_res['re'].append(re_tk_score)
    all_res['loc'].append(loc_tk_score)
    all_res['hop'].append(hop_tk_score)

    #if you want you can save for analyse

    # torch.save(all_res,f'./Fin_res/{args.setting}/cf.pkl') 

    with open('./data/MQuAKE-main/datasets/MQuAKE-CF.json', 'r') as f: 
        dataset = json.load(f)[:ds]
    gropu_data={}
    for d in dataset:
        d['ins']=''
        for rw in d['requested_rewrite']:
            d['ins']+=rw['prompt'].format(rw['subject'])+ ' '+rw['target_new']['str']+' '
        if len(d['single_hops']) not in     gropu_data:
            gropu_data[len(d['single_hops'])]=[d]
        else:
            gropu_data[len(d['single_hops'])].append(d)
    all_res={}
    for i,j in gropu_data.items():
        hop_score=[]
        hop_tk_score=[]
        all_res[i]=[]
        for ai,a in enumerate(j):
            temp=[]
            a['prompt_hop'] = a['questions'][0]
            portability_ans = a['new_answer']
            pro_input = PROMPT_DICT["only_ins_input_hop"].format_map(a)

            pro_res_token = test_prediction_acc(model, tokenizer, pro_input, portability_ans, 0)
            temp.append(pro_res_token)
            for aa in a['new_single_hops']:
                sub_input={'ins':a['ins'],'prompt_hop':aa['question']}
                sub_res_token = test_prediction_acc(model, tokenizer,  PROMPT_DICT["only_ins_input_hop"].format_map(sub_input), aa['answer'],0)

                temp.append(sub_res_token)
            hop_tk_score.append(temp)


        LOG.info(f'***Finally HOP res {len(j)} ***********')

        LOG.info(f'hop_tk_score-{i}: {sum([e[0][0] for e in hop_tk_score]) / len(hop_tk_score)}\n')


        LOG.info(f'hop-acc-{i}: {sum([ee[0] for e in hop_tk_score for ee in e[1:]]) / len([ee[0] for e in hop_tk_score for ee in e[1:]])}\n')

        all_res[i].append(hop_tk_score)
    fin_res=[]
    for i,j in all_res.items():
        for jj in j:

            for jjj in jj:

                fin_res.append([jjj[0],jjj[1:]])
    LOG.info(        f'Fin-acc-{i}: {sum([e[0][0] for e in fin_res ]) / len([e[0][0]  for e in fin_res ])}\n')
    LOG.info(        f'Fin-acc-hop-{i}: {sum([ee[0] for e in fin_res for ee in e[1]]) / len([ee[0]  for e in fin_res for ee in e[1]])}\n')


    torch.save(all_res,f'./Fin_res/{args.setting}/MQ.pkl')



    #print()
def get_mqT_res(model,tokenizer,LOG,ds=-1):
    with open('./data/datasets/MQuAKE-T.json', 'r') as f: #Muqake-T paths
        dataset = json.load(f)[:ds]
    embs, md, tk, nfs = make_mem(dataset)
    gropu_data={}
    for d in dataset:
        d['ins']=''

        for rw in d['requested_rewrite']:
            d['ins']+=rw['prompt'].format(rw['subject'])+ ' '+rw['target_new']['str']+' '
        if len(d['single_hops']) not in     gropu_data:
            gropu_data[len(d['single_hops'])]=[d]
        else:
            gropu_data[len(d['single_hops'])].append(d)
    all_res={}
    for i,j in gropu_data.items():
        hop_score=[]
        hop_tk_score=[]
        all_res[i]=[]
        for ai,a in enumerate(j):
            temp=[]
            a['prompt_hop'] = a['questions'][0]
            portability_ans = a['new_answer']
            subq=' '.join([aas['question'] for aas in a['new_single_hops']])
            fact_ids = retrieve_facts(a['prompt_hop']+subq, embs, md, tk, r=True)

            a['ins'] = ' '.join([nfs[fact_ids[1][iii]] for iii in range(i)])
            pro_input = PROMPT_DICT["only_ins_input_hop"].format_map(a)

            pro_res_token = test_prediction_acc(model, tokenizer, pro_input, portability_ans, 0)

            temp.append(pro_res_token)
            for aa in a['new_single_hops']:
                sub_input={'ins':a['ins'],'prompt_hop':aa['question']}
                sub_res_token = test_prediction_acc(model, tokenizer,  PROMPT_DICT["only_ins_input_hop"].format_map(sub_input), aa['answer'],0)

                temp.append(sub_res_token)
            hop_tk_score.append(temp)

        LOG.info(f'***Finally HOP res {len(j)} ***********')
        LOG.info(f'hop_tk_score-{i}: {sum([e[0][0] for e in hop_tk_score]) / len(hop_tk_score)}\n')
        all_res[i].append(hop_tk_score)

        LOG.info(f'hop_acc-{i}: {sum([ee[0] for e in hop_tk_score for ee in e[1:]]) / len([ee[0] for e in hop_tk_score for ee in e[1:]])}\n')

    fin_res = []
    for i, j in all_res.items():
        for jj in j:

            for jjj in jj:

                fin_res.append([jjj[0], jjj[1:]])
    LOG.info(f'Fin-acc-{i}: {sum([e[0][0] for e in fin_res]) / len([e[0][0] for e in fin_res])}\n')
    LOG.info(
        f'Fin-acc-hop-{i}: {sum([ee[0] for e in fin_res for ee in e[1]]) / len([ee[0] for e in fin_res for ee in e[1]])}\n')


    torch.save(all_res,f'./Fin_res/{args.setting}/MQT.pkl')
def get_zsre_res(model,tokenizer,LOG,ds=-1):

    ann = json.load(
        open('./data/zsre_mend_eval_portability_gpt4_ins.json'))[:ds] #ZsRE results

    log_dic = {}


    log_dic['edit'] = []
    log_dic['loc'] = []
    log_dic['re'] = []
    log_dic['hop'] = []


    loc_tk_score = []
    ed_tk_score=[]
    re_tk_score=[]
    hop_tk_score=[]
    retri_acc={
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }
    all_res = {
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }
    embs, md, tk, nfs = make_mem(ann)

    for ai, a in enumerate(ann):
        begin=time.time()
        try:
            a['rephrase_prompt'] = random.choice(a['paraphrase_prompts'])
            a['prompt'] = a['requested_rewrite']['prompt'].format(a['requested_rewrite']['subject'])
            a['locality_prompt'] = random.choice(a['neighborhood_prompts'])
            locality_ans = a['requested_rewrite']['target_true']['str']
            new_targe = a['requested_rewrite']['target_new']['str']
        except:
            a['rephrase_prompt'] = a['rephrase']
            a['prompt'] = a['src']
            a['locality_prompt'] = a['loc']
            locality_ans = a['loc_ans']
            new_targe = a['alt']
        a['oins'] = a['ins']
        # eval the edit
        fact_ids = retrieve_facts_r(a['prompt'], embs, md, tk, r=True)
        if torch.std(fact_ids[0]) > 0.1:
            a['ins'] = nfs[fact_ids[1][0]]
            edit_input = PROMPT_DICT["only_ins_input_ed"].format_map(a)
            ed_res_tk = test_prediction_acc(model, tokenizer, edit_input, new_targe, device=0,head=f'{ai}_ed_att')
            ed_tk_score.append(ed_res_tk)
        else:
            # ed_tk_score.append([0])
            ed_res_tk = test_prediction_acc(model, tokenizer, edit_input, new_targe, 0, locality=True, o_lc=a['prompt'])
            re_tk_score.append([ed_res_tk[1]])

        # # eval the rephrase
        fact_ids = retrieve_facts(a['rephrase_prompt'], embs, md, tk, r=True)
        if torch.std(fact_ids[0]) > 0.1:
            a['ins'] = nfs[fact_ids[1][0]]
            re_input = PROMPT_DICT["only_ins_input"].format_map(a)
            re_res_tk = test_prediction_acc(model, tokenizer, re_input, new_targe, 0,head=f'{ai}_re_att')
            re_tk_score.append(re_res_tk)
        else:
            # re_tk_score.append([0])
            re_res_tk = test_prediction_acc(model, tokenizer, re_input, new_targe, 0, locality=True,
                                            o_lc=a['rephrase_prompt'],head=f'{ai}_re_att')
            re_tk_score.append([re_res_tk[1]])

        # eval the loc
        fact_ids = retrieve_facts(a['locality_prompt'], embs, md, tk, r=True)
        if torch.std(fact_ids[0]) > 0.1:
            a['ins'] =nfs[fact_ids[1][0]]
            loc_input = PROMPT_DICT["only_ins_input_loc"].format_map(a)
            loc_res_tk = test_prediction_acc(model, tokenizer, loc_input, locality_ans, 0, locality=True,
                                             o_lc=a['locality_prompt'],head=f'{ai}_loc_att')
            loc_tk_score.append(loc_res_tk)
        else:
            a['ins']= a['oins']
            loc_input = PROMPT_DICT["only_ins_input_loc"].format_map(a)
            loc_res_tk = test_prediction_acc(model, tokenizer, loc_input, locality_ans, 0, locality=True,
                                             o_lc=a['locality_prompt'],head=f'{ai}_loc_att')
            loc_tk_score.append([1, 1, 1])

        # eval the hop
        if 'portability' in a:
            a['prompt_hop'] = a['portability']['New Question']
            portability_ans = a['portability']['New Answer']
            fact_ids = retrieve_facts(a['prompt_hop'], embs, md, tk, r=True)
            if torch.std(fact_ids[0]) > 0.1:
                a['ins'] = nfs[fact_ids[1][0]]

                pro_input = PROMPT_DICT["only_ins_input_hop"].format_map(a)
                pro_res_token = test_prediction_acc(model, tokenizer, pro_input, portability_ans, 0,head=f'{ai}_por_att')
                hop_tk_score.append(pro_res_token)
            else:

                pro_res_token = test_prediction_acc(model, tokenizer, pro_input, portability_ans, 0, locality=True,
                                                    o_lc=a['prompt_hop'],head=f'{ai}_por_att')
                hop_tk_score.append([pro_res_token[1]])
        end=time.time()
        LOG.info(f'Edits time {end-begin} ')

    LOG.info(f'********** Finlly zsre {args.setting} {len(ann)} ********** ')
    # LOG.info(f'zs_hop_score: {sum([e for e in hop_score]) / len(hop_score)}\n')
    LOG.info(f'zs_hop_tk_score: {sum([e[0] for e in hop_tk_score]) / len(hop_tk_score)}\n')
    # LOG.info(f'zs_edit_score: {sum([e for e in edit_score]) / len(edit_score)}\n')
    LOG.info(f'zs_ed_tk_score: { sum([e[0] for e in ed_tk_score]) / len(ed_tk_score)}\n')
    # LOG.info(f'zs_re_score: {sum([e for e in re_score]) / len(re_score)}\n')
    LOG.info(f'zs_re_tk_score: {sum([e[0] for e in re_tk_score]) / len(re_tk_score)}\n')
    # LOG.info(f'zs_loc_score: {sum([e for e in loc_score]) / len(loc_score)}\n')
    LOG.info(f'zs_loc_with_ins_tk_score: {sum([e[0] for e in loc_tk_score]) / len(loc_tk_score)}\n')
    LOG.info(f'zs_loc_with_pre_tk_score: {sum([e[1] for e in loc_tk_score]) / len(loc_tk_score)}\n')
    LOG.info(f'zs_loc_with_pre_trues_score: {sum([e[2] for e in loc_tk_score]) / len(loc_tk_score)}\n')
    LOG.info(f'zs_loc_with_re_and_trues: {sum([max(e[2],e[1],e[0]) for e in loc_tk_score]) / len(loc_tk_score)}\n')

    all_res['ed'].append(ed_tk_score)
    all_res['re'].append(re_tk_score)
    all_res['loc'].append(loc_tk_score)
    all_res['hop'].append(hop_tk_score)

    torch.save(all_res,f'./Fin_res/{args.setting}/zs.pkl')




def get_rever_res(model,tokenizer,LOG,ds=-1):

    ann = json.load(
        open('./data/portability/Inverse Relation/zsre_inverse_relation.json'))[:ds]

    log_dic = {}


    log_dic['edit'] = []
    log_dic['loc'] = []
    log_dic['re'] = []
    log_dic['hop'] = []


    loc_tk_score = []
    ed_tk_score=[]
    re_tk_score=[]
    hop_tk_score=[]
    retri_acc={
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }
    all_res = {
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }

    for ai, a in enumerate(ann):


        a['rephrase_prompt'] = a['rephrase']
        a['prompt'] = a['src']
        a['locality_prompt'] = a['inverse question']
        inver_ans = a['subject']
        new_targe = a['alt']
        a['ins']=a['cond']
        # eval the edit
        edit_input = PROMPT_DICT["only_ins_input_ed"].format_map(a)
        ed_res_tk = test_prediction_acc(model, tokenizer, edit_input, new_targe, 0)
        ed_tk_score.append(ed_res_tk)
        # # eval the rephrase
        re_input = PROMPT_DICT["only_ins_input"].format_map(a)
        re_res_tk = test_prediction_acc(model, tokenizer, re_input, new_targe, 0)
        re_tk_score.append(re_res_tk)
        # eval the loc
        loc_input = PROMPT_DICT["only_ins_input_loc"].format_map(a)
        loc_res_tk = test_prediction_acc(model, tokenizer, loc_input, inver_ans,  0)
        loc_tk_score.append(loc_res_tk)

        LOG.info(f'********** Finlly reverse {args.setting} {len(ann)} ********** ')
        LOG.info(f'zs_ed_tk_score: { sum([e[0] for e in ed_tk_score]) / len(ed_tk_score)}\n')
        LOG.info(f'zs_re_tk_score: {sum([e[0] for e in re_tk_score]) / len(re_tk_score)}\n')
        LOG.info(f'zs_inver_with_ins_tk_score: {sum([e[0] for e in loc_tk_score]) / len(loc_tk_score)}\n')


    all_res['ed'].append(ed_tk_score)
    all_res['re'].append(re_tk_score)
    all_res['loc'].append(loc_tk_score)
    all_res['hop'].append(hop_tk_score)

    torch.save(all_res,f'./alb_eval/{args.setting}/zs.pkl')

def get__replace(model,tokenizer,LOG,ds=-1,dt=None):
    if dt is None:
        ann = json.load( open('./data/portability/Subject Replace/counterfact_subject_replace.json'))[:ds]
    else:
        ann = json.load( open('./data/portability/Subject Replace/zsre_subject_replace.json'))[:ds]

    log_dic = {}


    log_dic['edit'] = []
    log_dic['loc'] = []
    log_dic['re'] = []
    log_dic['hop'] = []


    loc_tk_score = []
    ed_tk_score=[]
    re_tk_score=[]
    hop_tk_score=[]
    retri_acc={
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }
    all_res = {
        'ed': [],
        're': [],
        'loc': [],
        'hop': [],
    }

    for ai, a in enumerate(ann):

        try:
            a['rephrase_prompt'] = random.choice(a['paraphrase_prompts'])
            a['prompt'] = a['requested_rewrite']['prompt'].format(a['requested_rewrite']['subject'])
            a['locality_prompt'] = a['requested_rewrite']['prompt'].format(a['alternative_subject'])

   
            new_targe = a['requested_rewrite']['target_new']['str']
            a['sub_ins']=a['requested_rewrite']['subject']+ ' also known as '+a['alternative_subject']
            a['ins'] =a['sub_ins'] +'. ' + a['prompt'] +' '+new_targe
        except:
            a['rephrase_prompt'] = a['rephrase']
            a['prompt'] = a['src']
            a['locality_prompt'] = a['alter_subject_question']
    
            new_targe = a['alt']
            a['sub_ins'] = a['subject'] + ' also known as ' +a['alternative_subject']
            a['ins']= a['sub_ins']+'. '+ a['src']+ ' '+ a['alt']

        # eval the edit
        edit_input = PROMPT_DICT["only_ins_input_ed"].format_map(a)
        ed_res_tk = test_prediction_acc(model, tokenizer, edit_input, new_targe, 0)
        ed_tk_score.append(ed_res_tk)
        # # eval the rephrase
        re_input = PROMPT_DICT["only_ins_input"].format_map(a)
        re_res_tk = test_prediction_acc(model, tokenizer, re_input, new_targe, 0)
        re_tk_score.append(re_res_tk)
        # eval the loc
        loc_input = PROMPT_DICT["only_ins_input_loc"].format_map(a)
        loc_res_tk = test_prediction_acc(model, tokenizer, loc_input, new_targe,  0)
        loc_tk_score.append(loc_res_tk)


        LOG.info(f'********** Finlly replace  {args.setting} { ai} \{len(ann)} ********** ')
        LOG.info(f'zs_ed_tk_score: { sum([e[0] for e in ed_tk_score]) / len(ed_tk_score)}\n')
        LOG.info(f'zs_re_tk_score: {sum([e[0] for e in re_tk_score]) / len(re_tk_score)}\n')
        LOG.info(f'zs_replace_with_ins_tk_score: {sum([e[0] for e in loc_tk_score]) / len(loc_tk_score)}\n')


    all_res['ed'].append(ed_tk_score)
    all_res['re'].append(re_tk_score)
    all_res['loc'].append(loc_tk_score)
    all_res['hop'].append(hop_tk_score)

    torch.save(all_res,f'./alb_eval/{args.setting}/zs.pkl')


def init_log():
    uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

    file_name = f'{args.setting}'
    log_path = f'./Fin_res/{file_name}/' #log paths
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    LOG = logging.getLogger(__name__)
    LOG.setLevel(level=logging.INFO)
    fh = logging.FileHandler(log_path + 'log.txt')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    LOG.addHandler(fh)
    LOG.addHandler(ch)
    LOG.info(f'LOG path is {log_path}')
    LOG.info(f'setting is {args}')
    return LOG




LOG = init_log()

model,tokenizer=load_models()

get_zsre_res(model,tokenizer,LOG,ds=-1)
get_ann_res(model,tokenizer,LOG,ds=-1)
get_mqT_res(model,tokenizer,LOG,ds=-1)
get_rever_res(model,tokenizer,LOG,ds=-1)
get__replace(model,tokenizer,LOG,ds=-1)
print('DONE',args.setting)

