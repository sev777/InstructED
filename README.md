


This codebase contains the implementation of the paper [InstructEd:    Soft-Instruction Tuning for     Model Editing with Hops] in proceedings of the [ACL 2024 Findings]conference.


## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt

Download the Transformer to local, and replace the modeling_gpt2.py and modeling_llama.py with the provide version.

mv ./transformers_rep/models ./transformers/models
```


Finetuning the models
```bash
python -u  finetuning.py --llama_model_path you_root_path/llama2_7B --train_data_path ./data/counterfact-train_ins.json --val_data_path ./data/counterfact-val_ins.json --adapter_type no_gate --output_dir ./results --batch_size 6 --model Llama7B_adapter --adapter_st 10 --adapter_ed 20
```
You will get the results on ./results named checkpoint-*.pth.


Evaluate the models
```bash
python -u evals_with_ret.py --adapter_path your_results_path/checkpoint-9.pth --adapter_type no_gate --model_tp /open_llama_7b --setting your_logs_name
```



