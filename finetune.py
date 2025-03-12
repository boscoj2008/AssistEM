import math
import wandb
#import fire
import torch
import logging
from torch.utils.data import DataLoader
from argparse import(Namespace, 
ArgumentParser
)
from transformers import(
AutoModelForCausalLM, 
AutoTokenizer, 
get_scheduler,
AutoConfig
)
from peft import(
LoraConfig, 
IA3Config,
prepare_model_for_kbit_training, 
get_peft_model,
TaskType
)
from utils import get_bnb_config, NEFTune
from src.dataset import(
InstructionDataset, 
collate_func,
generate_data
)
from src.optimizer import get_optimizer
from src.trainer import Trainer
from src.utils import(
set_random_seeds 
)
from datasets import load_dataset
import os

def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="AssistEM Instruction Tuning")
    # huggyllama/llama-7b
    parser.add_argument("--base_model_path", type=str,
                        default="yahma/llama-7b-hf", 
                        help="foundation model to use.")
    parser.add_argument("--train_data_path", type=str,
                        default="dataset/train.json",
                        help="Path to train data.")
    parser.add_argument("--valid_data_path", type=str,
                        default="dataset/valid.json",
                        help="Path to validation data.")
    parser.add_argument("--valid_percent", type=float,
                        default=0.02,
                        help="% valid")
    parser.add_argument("--batch_size", type=int,
                        default=16,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=1,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=2,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=2e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=0,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="constant",
                        help="learning rate scheduler: linear, constant, cosine, cosine_warmup")
    parser.add_argument("--warm_up_step", type=int,
                        default=0,
                        help="number of warm up steps")
    parser.add_argument("--lora_rank", type=int,
                        default=16,
                        help="rank of lora")
    parser.add_argument("--peft_type", type=str,
                        default="LoRA",
                        help="select peft type of choice: LoRA, QLoRA")  
    parser.add_argument("--linear_layers", action="store_true",
                        help="target LoRA or QLoRA linear layers else attention blocks ") 
    parser.add_argument("--flash_attention", action='store_true',
                        help="speedup with flash attention")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="use grad checkpointing to avoid OOM error")                        
    parser.add_argument("--input_type", type=str, default="BrandTitle",
                        help="BrandTitle, Title or BrandTitlePrice")                        
    parser.add_argument("--max_val_seq_len", type=int,
                        default=4096,  help="max_val_seq_len")                                                 
                        
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()
    args = parse_arguments()
    print(''.join(f'{k}={v}\n' for k, v in vars(args).items()))
    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True, trust_remote_code=True)
    train_data, valid_data = generate_data(args.train_data_path, valid_path=args.valid_data_path, valid_size =args.valid_percent) 
    
    # check for 'falcon' or 'mpt' in the tokenizer's name_or_path
    if 'falcon' in tokenizer.name_or_path:
    #https://gist.github.com/pacman100/1731b41f7a90a87b457e8c5415ff1c14#file-falcon_peft-py-L173
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    elif 'galactica' in tokenizer.name_or_path:   
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1

    elif 'internlm2' in tokenizer.name_or_path or 'bloom' in tokenizer.name_or_path or 'opt' in tokenizer.name_or_path:   
        pass   
        
    else:
        tokenizer.pad_token_id = 0   
    
    train_dataset = InstructionDataset(train_data, tokenizer, max_length=184, input_type=args.input_type)
    valid_dataset = InstructionDataset(valid_data, tokenizer, max_length=args.max_val_seq_len, input_type=args.input_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)

    # Prepare model & config
    #config = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True)
    #config.attn_config['attn_impl'] = 'triton' if 'mpt'in args.base_model_path else config
    
    if args.peft_type == 'QLoRA':
    #Note that FlashAttention can only be used for models with the fp16 or bf16 torch type, so make sure to cast your model to the appropriate type before using it.
    #https://huggingface.co/docs/transformers/perf_infer_gpu_one#expected-speedups
        bnb_config = get_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
           args.base_model_path,
           torch_dtype=torch.bfloat16,
           #use_flash_attention_2 = args.flash_attention, # supports llama, falcon and whisper
           quantization_config=bnb_config,
           trust_remote_code=True
            )
    
    else:
    #Note that FlashAttention can only be used for models with the fp16 or bf16 torch type, so make sure to cast your model to the appropriate type before using it.
    #https://huggingface.co/docs/transformers/perf_infer_gpu_one#expected-speedups
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            #use_flash_attention_2 = args.flash_attention, # supports llama, falcon and whisper
            load_in_8bit =True,
            device_map = 'auto',
            trust_remote_code=True
            )
    assert args.peft_type in ['LoRA', 'QLoRA'], "Please choose either 'IA3' or 'LoRA'/'QLoRA' for args.peft_type"
    if args.peft_type == "QLoRA" or args.peft_type == "LoRA":
        if args.linear_layers:        
            # target linear layers 
                if model.config.model_type == 'llama' or model.config.model_type == 'mistral' or 'deci' in model.config.model_type:        
                    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']

                elif model.config.model_type == 'falcon':
                    target_modules = ['dense_4h_to_h','query_key_value','dense_h_to_4h', 'dense'] 
                    model.config.use_cache = False
                    
                elif model.config.model_type == 'internlm2':
                    target_modules= ['wqkv', 'wo', 'w1', 'w2', 'w3' ]

                elif model.config.model_type == 'bloom':
                    target_modules = ["dense_h_to_4h","dense_4h_to_h", "dense" ,"query_key_value"] 
  
                elif 'opt' in model.config.model_type:
                    target_modules = ['k_proj', 'v_proj', 'q_proj', 'fc1', 'fc2', 'out_proj']   
                    
                elif 'mpt' in model.config.model_type:                    
                    target_modules = ["Wqkv","up_proj", "down_proj", "out_proj"]         
                    args.use_gradient_checkpointing = False
 

        else:
            # targeet attention blocks
            target_modules = ["q_proj", "v_proj"] 
    elif args.peft_type ==  "IA3":
        target_modules =  ["k_proj", "v_proj","q_proj", "down_proj", 'dense', 'lm_head'] 
        feedforward_modules=["down_proj"]                   
              

    peft_config = (
        LoraConfig(
            lora_alpha=8,
            lora_dropout=0.10,
            r=args.lora_rank,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        if args.peft_type == 'LoRA' or args.peft_type == 'QLoRA'
        else IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules
        )
    )
    #model = NEFTune(model, noise_alpha=5)
    # https://huggingface.co/docs/peft/package_reference/peft_model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = args.use_gradient_checkpointing) # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    model = get_peft_model(model, peft_config)
    # print model % trainable params
    model.print_trainable_parameters() 
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warm_up_step / args.accum_grad_step),
        num_training_steps=max_train_steps,
    )

    # Prepared logger
    wandb.init(
        project="AssitEM",
        name="fine-tune" + f"_{os.path.basename(model.config._name_or_path)}", 
        config={
            "tokenizer": args.base_model_path,
            "model": args.base_model_path,
            "epoch": args.epoch,
            "train_data_num": args.valid_percent,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warm_up_step": args.warm_up_step,
            "lora_rank": args.lora_rank,
            #"run_no": args.run_no
        }
    )
    wandb.watch(model, log="all")

    # Start training
    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        accum_grad_step=args.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
        peft_type=args.peft_type
    )
    trainer.fit(epoch=args.epoch)        
