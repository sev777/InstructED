import random

from torch.utils.data import Dataset
import json
import torch
import copy
import  random

from transformers import AutoTokenizer, AutoModel

PROMPT_DICT = {
    "ins_prompt":(

        "Instruction:\n{ins}\nInput:\n{questions[0]}"
    ),
    "only_ins_input_hop": (
        "Instruction:\n{ins}\nInput:\n{questions[0]}"
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


class InstructionDataset(Dataset):
    def __init__(self, data_path, model_path, max_words=50, Tokenizer=None, partition="train"):
        self.ann = json.load(open(data_path))
        self.lann=[a for a in self.ann if len(a)==9]
        self.retri='/root/siton-data-9aa46a4f0e354f65bd8679947a35e67e/LMs/LMs/huggingface/contriever-msmarco'# your contriever-msmarco path
        self.ret_tok = AutoTokenizer.from_pretrained(self.retri)
        self.ret = AutoModel.from_pretrained(self.retri).cuda()

        print(f'****************{partition}****************')
        print(self.ann[:2])
        print(f'****************************************')



        self.max_words = max_words
        self.tokenizer1 = Tokenizer
        self.cos=torch.nn.CosineSimilarity()
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        ann['ins']= ann['ins'].replace('\n','')
        #Differet dataset have different keys. So add try... excpet...
        try:
            ann['rephrase_prompt'] = ann['rephrase_prompt'].replace('\n', '')
            ann['locality_prompt'] = ann['locality_prompt'].replace('\n', '')
            #
            input_prompt = PROMPT_DICT["only_ins_input"].format_map(ann)
            input_example = input_prompt +' ' + ann["target_new"].replace('\n','')

            loc_prompt = PROMPT_DICT["only_ins_input_loc"].format_map(ann)
            loc_example = loc_prompt +' '+ ann["locality_ground_truth"].replace('\n','')
            loc_dt = ann['locality_prompt']
            loc_ex = ann['locality_prompt'] + ' ' + ann["locality_ground_truth"].replace('\n', '')

        except:

            ann['rephrase_prompt'] = ann['questions'][0].replace('\n', '')
            lc_data=random.choice(self.lann)
            ann['locality_prompt'] = lc_data['locality_prompt'].replace('\n', '')

            input_prompt = PROMPT_DICT["only_ins_input"].format_map(ann)
            input_example = input_prompt +' ' + ann["new_answer"].replace('\n','')

            loc_prompt = PROMPT_DICT["only_ins_input_loc"].format_map(ann)
            loc_example = loc_prompt +' '+lc_data["locality_ground_truth"].replace('\n','')
            loc_dt=  ann['locality_prompt']
            loc_ex= ann['locality_prompt']+' '+lc_data["locality_ground_truth"].replace('\n','')



        return input_prompt,input_example,loc_prompt,loc_example,loc_dt,loc_ex, ann['rephrase_prompt'],ann['ins']


    def encoder_inpt(self,prompt,example,max_length):

        prompt = torch.tensor(self.tokenizer1.encode(prompt,), dtype=torch.int64)
        example = torch.tensor(self.tokenizer1.encode(example,), dtype=torch.int64)

        padding = max_length - example.shape[0]

        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: max_length]


        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()



        return example,labels,example_mask,label_mask

    def collate_fn(self, batch):

        max_length=max( len(self.tokenizer1.encode(bb)) for b in batch for bb in b)
        int_example=[ self.encoder_inpt(b[0],b[1],max_length)  for b in batch]
        loc_example=[self.encoder_inpt(b[2],b[3],max_length) for b in batch]
        loc_ori_example=[self.encoder_inpt(b[4],b[5],max_length) for b in batch]

        keys=['example','labels','example_mask','label_mask']

        inputs={}
        for i in  int_example+loc_example:

            for ki, k in enumerate(keys):
                if k not in inputs:
                    inputs[k]=[i[ki]]
                else:
                    inputs[k].append(i[ki])

        inputs={i:torch.stack(j)  for i,j in inputs.items()}
        assert all(inputs['example'][inputs['labels']>0]==inputs['labels'][inputs['labels']>0])


        ins_lens= [len(self.tokenizer1.encode('Instruction:\n' + b[-1])) for b in batch]*2
        sims=[        self.retrieve_facts(b[-1],b[-2],self.ret,self.ret_tok) for b in batch]
        lcsims=[        self.retrieve_facts(b[-1],b[-4],self.ret,self.ret_tok) for b in batch]
        batch.append([ins_lens,sims+lcsims])
   

        lc_inputs={}


        for i in loc_ori_example:

            for ki, k in enumerate(keys):
                if k not in lc_inputs:
                    lc_inputs[k] = [i[ki]]
                else:
                    lc_inputs[k].append(i[ki])

        lc_inputs = {i: torch.stack(j) for i, j in lc_inputs.items()}


        return {'inputs':inputs,'lc_inputs':lc_inputs,'ori':batch}

    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def retrieve_facts(self,query, fact, contriever, tok, k=1):
        inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
        fact_embs = tok([fact], padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            query_emb = self.mean_pooling(outputs[0], inputs['attention_mask']).cpu()
            outputs = contriever(**fact_embs)
            fact_embs = self.mean_pooling(outputs[0], fact_embs['attention_mask']).cpu()

        sim = (query_emb @ fact_embs.T)[0].tolist()[0]
        cos_sim=self.cos(query_emb, fact_embs).tolist()[0]

        return sim,cos_sim

    def encoder_ed_inpt(self,prompt,example):

        prompt = torch.tensor(self.tokenizer1.encode(prompt,), dtype=torch.int64)
        example = torch.tensor(self.tokenizer1.encode(example,), dtype=torch.int64)

        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]


        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1


        label_mask = labels.ge(0)
   
        labels[~label_mask] = 0

        label_mask = label_mask.float()

        padding = self.max_words - prompt.shape[0]
        if padding > 0:
            prompt = torch.cat(( torch.zeros(padding, dtype=torch.int64) - 1,prompt))
        elif padding < 0:
            prompt = prompt[: self.max_words]
        prompt_mask = prompt.ge(0)
        prompt[~prompt_mask] = 0


        return prompt,labels,prompt_mask,label_mask
