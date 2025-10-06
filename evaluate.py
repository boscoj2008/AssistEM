

import logging
from tqdm import tqdm
from argparse import Namespace, ArgumentParser
import wandb
import torch
import json
#import thunder
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from utils import get_bnb_config, update_json_result
from src.dataset import generate_test_data, generate_data, collate_func
from src.utils import set_random_seeds, dict_to_device, save_json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time 
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="AssistEM Evaluation")
    parser.add_argument("--method", type=str,
                        default="lora-fine-tune",
                        help="support method: zero-shot, two-shot, and three-shot")
    parser.add_argument("--base_model_path", type=str,
                        default="yahma/llama-7b-hf",
                        help="Path to the checkpoint of LlaMA "
                        )
    parser.add_argument("--peft_path",type=str,
                        default="checkpoint/qlora",
                        help="Path to the saved PEFT checkpoint.")
    parser.add_argument("--test_data_path", type=str,
                        default="dataset/test/test-capped.json",
                        help="Path to test data.")                      
    parser.add_argument("--output_path", type=str,
                        default="prediction.json",
                        help="output path")
    parser.add_argument("--peft_type", type=str,
                        default="LoRA",
                        help="peft type")   
    parser.add_argument("--input_type", type=str, default="BrandTitle",
                        help="BrandTitle, Title or BrandTitlePrice")                                                

    parser.add_argument("--result_filename", type=str, default="results_mojo.json",
                        help="filename to store eval results")  

    parser.add_argument("--max_val_seq_len", type=int,
                        default=4096,  help="max_val_seq_len")    
    parser.add_argument("--train_set_size", type=int,
                        default=6000,  help="eval_type")    
                       
                                                
                        
    return parser.parse_args()

 
if __name__ == "__main__":
    set_random_seeds()
    
    args = parse_arguments()
    wandb.init(
        project="AssitEM",
        name=args.method,
     )

    logger = logging.getLogger("AssistEM Evaluation")
    print(''.join(f'{k}={v}\n' for k, v in vars(args).items()))
    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True, trust_remote_code=True)
    test_data = generate_data(data_path=args.test_data_path, train_val=False)
    
    # check for 'falcon' or 'mpt' in the tokenizer's name_or_path
    if 'falcon' in tokenizer.name_or_path:
    #https://gist.github.com/pacman100/1731b41f7a90a87b457e8c5415ff1c14#file-falcon_peft-py-L173
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    elif 'Qwen' in tokenizer.name_or_path:   
        tokenizer.bos_token_id = tokenizer.eos_token_id

    elif 'internlm2' in tokenizer.name_or_path or 'bloom' in tokenizer.name_or_path or  'opt' in tokenizer.name_or_path:      
        pass   
        
    else:
        tokenizer.pad_token_id = 0       
    
    
    test_dataset = generate_test_data(args, test_data=test_data, tokenizer = tokenizer)            
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)
    
    # Prepare model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    bnb_config = get_bnb_config()

    if args.peft_type == 'QLoRA':
        model = AutoModelForCausalLM.from_pretrained(
           args.base_model_path,
           torch_dtype=torch.bfloat16,
           quantization_config=bnb_config,
           trust_remote_code=True
            )
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit =True,
            device_map = 'auto',
            trust_remote_code=True
            )
    model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()

    preds = list()
    ground_truth = list()
    
    for instance in test_data:
        if instance["answer"] == "yes":
            ground_truth.append(1)
        else:
            ground_truth.append(0)    
    
    prediction_list = []
    test_bar = tqdm(test_loader, desc=f"Testing")
    start_time = time.process_time()
    for _, batch_data in enumerate(test_bar, start=1):
        with torch.no_grad():
            batch_data = dict_to_device(batch_data, device)
            generated_tokens = model.generate(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_new_tokens=50,
                do_sample=False, 
                top_p=0.9, 
                top_k=40,
                num_beams=2,
                output_scores=True,
                temperature=0.0
            )
           
            generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            generations = generations.replace(batch_data["prompt"][0], "").strip()
            if "yes" in generations.lower():
                preds.append(1)
            elif "no" in generations.lower():
                preds.append(0)
            else:
                preds.append(0) # not clear response    
                    
            prediction_list.append({"input": batch_data["prompt"][0], "output": generations})
            
            logger.debug(f"Question:\n{batch_data['prompt'][0]}\n")
            logger.debug(f"Answer:\n{generations}\n")
    delta = time.process_time() - start_time
    precision = precision_score(ground_truth, preds)
    recall = recall_score(ground_truth, preds)
    f1_score_ = f1_score(ground_truth, preds)
    accuracy = accuracy_score(ground_truth, preds)
    task_name = f'{os.path.basename(args.test_data_path)}'
    results = {
    "Acc": accuracy,
    "Precision": precision,
    "Recall": recall,
    "f1": f1_score_,
    "eval_time" : delta,
    "task_name": task_name,
    "model": model.config._name_or_path,
    "method": args.method,
    "data_size": args.train_set_size
    }
    update_json_result(args.result_filename, results) 
    
    print(results)

    with open(f'{os.path.dirname(args.test_data_path)}/f1_recall_results-{args.method}-{os.path.basename(args.peft_path)}.json', 'w') as f:  
        json.dump(results , f, indent=2)
    

    save_json(prediction_list, f'{args.method}_' + args.output_path)
