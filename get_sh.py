import os
import numpy as np

import pandas as pd
def get_sh():
    '''


    parser.add_argument("--input_file", default="Llama7B_adapter", type=str, help="Name of model to train")
    parser.add_argument("--output_file", default="Llama7B_adapter", type=str, help="Name of model to train")
    parser.add_argument("--adapter_path",
                        default="/root/siton-data-hanxiaoqiData/Ins_edit/output_dir/hop_withnogate_v2/checkpoint-8.pth",
                        type=str, help="Name of model to train")
    parser.add_argument("--layer_st", default=10, type=int, help="Name of model to train")
    parser.add_argument("--layer_ed", default=20, type=int, help="Name of model to train")
    parser.add_argument("--adapter_type", default="FFN", type=str, help="Name of model to train")
    parser.add_argument("--model_tp", default="/open_llama_7b_v2", type=str, help="Name of model to train")
    '''

    bsae_path='./attention_analyse/'
    for v in ['/open_llama_7b','/open_llama_7b_v2']:
        for adapter_type in ['FFN','no_gate','soft_gate','gate']:
            for ck in ['6','8','9']:
                if 'v2' in v:
                    # adapter_path=bsae_path+'no_hop_with_'+adapter_type+'_v2'+f'/checkpoint-{ck}.pth'
                    # print(
                    # f'\n CUDA_VISIBLE_DEVICES=1 python evals.py --adapter_path {adapter_path} --adapter_type {adapter_type} --model_tp {v} --setting lc_no_hop_{v[1:] + "_" + adapter_type + "_ck_" + ck}')

                    adapter_path = bsae_path + 'retri_hop_with_' + adapter_type + '_v2' + f'/checkpoint-{ck}.pth'
                    print(
                        f'\n CUDA_VISIBLE_DEVICES=0 python -u evals_with_ret.py --adapter_path {adapter_path} --adapter_type {adapter_type} --model_tp {v} --setting retri_hop_{v[1:] + "_" + adapter_type + "_ck_" + ck}')

                    # adapter_path = bsae_path + 'FTdt_hop_with_' + adapter_type + '_v2' + f'/checkpoint-{ck}.pth'
                    # print(
                    #     f'\n CUDA_VISIBLE_DEVICES=1 python evals.py --adapter_path {adapter_path} --adapter_type {adapter_type} --model_tp {v} --setting lcft_huo_{v[1:] + "_" + adapter_type + "_ck_" + ck}')

                else:
                    # adapter_path = bsae_path + 'no_hop_with_' + adapter_type+'_v1'+f'/checkpoint-{ck}.pth'
                    # print(f'\n CUDA_VISIBLE_DEVICES=0 python evals.py --adapter_path {adapter_path} --adapter_type {adapter_type} --model_tp {v} --setting lc_no_hop_{v[1:]+"_"+adapter_type+"_ck_"+ck}')

                    adapter_path = bsae_path + 'retri_hop_with_' + adapter_type+f'/checkpoint-{ck}.pth'
                    print(f'\n CUDA_VISIBLE_DEVICES=0 python -u evals_with_ret.py --adapter_path {adapter_path} --adapter_type {adapter_type} --model_tp {v} --setting retri_hop_{v[1:]+"_"+adapter_type+"_ck_"+ck}')

                    # adapter_path = bsae_path + 'FTdt_hop_with_' + adapter_type+'_v1'+f'/checkpoint-{ck}.pth'
                    # print(f'\n CUDA_VISIBLE_DEVICES=0 python evals.py --adapter_path {adapter_path} --adapter_type {adapter_type} --model_tp {v} --setting FT_huo_with_{v[1:]+"_"+adapter_type+"_ck_"+ck}')


                # '/root/siton-data-hanxiaoqiData/Ins_edit/results_dir/huo_with_gate_v1/checkpoint-9.pth'

# get_sh()

def extract_res():

    files=[]
    for i,j ,k in os.walk('/root/siton-data-hanxiaoqiData/Ins_edit/ori_evals'):
        if 'log.txt'  in k :
            files.append(i+'/log.txt')
        print()
        pass
    alls=[]
    for f in sorted(files):
        # try:
        if 1:
            data=open(f,'r').readlines()
            indexs=[di  for di,d in enumerate(data) if 'Fin' in d and 'acc' not in d and di!=0]
            if len(indexs)==2:
                lc_ins=  [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[0]:indexs[0] + 15] if d != '\n' and 'INFO' in d]
                lc_pre = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[1]:indexs[1] + 15] if d != '\n' and 'INFO' in d]

                # temp.append(ct[0][0])
                mat = lc_ins[1:] + lc_pre[1:]
                keys = [m[0] for m in mat]
                value = [m[1] for m in mat]
                if alls == []:
                    alls.append(['Type'] + keys)

                alls.append([f.split('/')[-2]] + value)
            if len(indexs)==3:
                hp1 = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[0]:indexs[0] + 2] if d != '\n']
                hp2 = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[1]:indexs[1] + 2] if d != '\n']
                hp3 = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[2]:indexs[2] + 2] if d != '\n']

                # temp.append(ct[0][0])
                mat = hp1[1:] + hp2[1:] + hp3[1:]
                keys = [m[0] for m in mat]
                value = [m[1] for m in mat]
                if alls == []:
                    alls.append(['Type'] + keys)

                alls.append([f.split('/')[-2]] + value)
            if len(indexs)==6:
                zsct = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[0]:indexs[0] + 9] if d != '\n']
                ct = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[1]:indexs[1] + 9] if d != '\n']

                hp1 = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[2]:indexs[2] + 2] if d != '\n']
                hp2 = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[3]:indexs[3] + 2] if d != '\n']
                hp3 = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[4]:indexs[4] + 2] if d != '\n']
                hp4 = [d.split('INFO - ')[-1].strip().split(':') for d in data[indexs[5]:] if d != '\n']

                # temp.append(ct[0][0])
                mat = zsct[1:]+ct[1:] + hp1[1:] + hp2[1:] + hp3[1:] + hp4[1:]
                keys = [m[0] for m in mat]
                value = [m[1] for m in mat]
                if alls == []:
                    alls.append(['Type'] + keys)

                alls.append([ct[0][0].replace('*', '')] + value)
            if len(indexs)==8:
                zs_res=  [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[0]:indexs[0] + 15] if d != '\n' and 'INFO' in d]
                ct_res = [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[1]:indexs[1] + 15] if d != '\n'and 'INFO' in d]
                hp2 = [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[2]:indexs[2] + 4] if d != '\n'and 'INFO' in d]
                hp3 = [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[3]:indexs[3] + 4] if d != '\n'and 'INFO' in d]
                hp4 = [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[4]:indexs[4] + 8] if d != '\n'and 'INFO' in d]
                hp2t = [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[5]:indexs[5] + 4] if d != '\n'and 'INFO' in d]
                hp3t = [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[6]:indexs[6] + 4] if d != '\n'and 'INFO' in d]
                hp4t = [d.split('INFO - ')[-1].strip().replace('\r','').split(':') for d in data[indexs[7]:indexs[7] + 11] if d != '\n'and 'INFO' in d]

                # temp.append(ct[0][0])
                mat = zs_res[1:] + ct_res[1:] + hp2[1:] + hp3[1:] + hp4[1:] + hp2t[1:] + hp3t[1:] + hp4t[1:]
                keys = [m[0] for m in mat  ]
                value = [m[1] for m in mat]
                if alls == []:
                    alls.append(['Type'] + keys)

                alls.append([f.split('/')[-2].replace('\r','')] + value)

        # except:
        #     print(f)
        #     continue
        # hp1=data[4805:4810]
        # hp2=data[8555:8560]
        # hp3=data[8942:9847]
        # hp4=data[10283:]


        print()
    pd.DataFrame(alls[1:],columns=alls[0]).to_csv('./orievals.csv')
extract_res()

# import json
#
# data=json.load(open('./data/counterfact-train_ins.json'))
# # edata=json.load(open('./data/zsre_mend_eval_ins.json'))

print()

#12200
#10000 for recall
#2200 multi  {1: 0, 2: 1000, 3: 910, 4: 290}



