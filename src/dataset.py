import torch
from torch.utils.data.dataset import Dataset
from utils import entity_matching_prompt
from datasets import load_dataset

class InstructionDataset(Dataset):
    
    def __init__(
                 self, 
                 data_list, 
                 tokenizer, 
                 input_type,
                 max_length=512, 
                 is_train=True, 
                 incontext=False,
                 num = 2
                ):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.incontext = incontext
        self.num = num
        self.input_type = input_type
        self.data_list = self.transform(list(data_list))

    def pad_or_truncate(
                        self, 
                        data, 
                        padding_token=0
                       ):
        if self.max_length >= len(data):
            return data + [padding_token] * (self.max_length - len(data))
        else:
            return data[:self.max_length]

        
    def transform(
                  self, 
                  data_list
                 ):
        
        #if 'tiiuae/falcon-7b' in self.tokenizer.name_or_path:
        #    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #    self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        #else:
        #    self.tokenizer.pad_token_id = 0
 
        
        # instructions 
        instructions = [entity_matching_prompt(x, exmpls=self.incontext, num=self.num, input_type=self.input_type) for x in data_list]
        tokenized_instructions = self.tokenizer(instructions, add_special_tokens=False)

        processed_data = []
        if self.is_train:            
            outputs = [x["output"] for x in data_list]
            tokenized_outputs = self.tokenizer(outputs, add_special_tokens=False)

            for i in range(len(data_list)):
                instructions_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]  # inst & input          
                outputs_ids = tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id] # target
                processed_data_input_ids =  instructions_input_ids + outputs_ids
                processed_data.append(
                    {
                        "input_ids": self.pad_or_truncate(processed_data_input_ids, padding_token = self.tokenizer.pad_token_id),
                        "attention_mask": self.pad_or_truncate([1] * len(processed_data_input_ids), padding_token = self.tokenizer.pad_token_id),
                        "labels": self.pad_or_truncate([-100] * len(instructions_input_ids) + outputs_ids, padding_token = self.tokenizer.pad_token_id),
                        "output_mask": self.pad_or_truncate([0] * len(instructions_input_ids) + [1] * len(outputs_ids), padding_token = self.tokenizer.pad_token_id),
                    }
                )
        else:
            for i in range(len(data_list)):
            
                processed_data_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]  # bos + query + zrs-context            
                processed_data.append(
                    {
                        "input_ids": processed_data_input_ids,
                        "attention_mask": [1] * len(processed_data_input_ids),
                        "prompt": instructions[i],
                    }
                )
        return processed_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def collate_func(data: list) -> dict:
    # convert list of dict to dict of list
 
    data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}
    #print(data_list_dict)
    # convert dict of list to dict of torch tensor
    data_tensor_dict = {
        k: v if isinstance(v[0], str) else torch.tensor(v)
        for k, v in data_list_dict.items()
    }
    return data_tensor_dict
    
    
    
    
def generate_test_data(args, test_data, tokenizer, exemplars=None):
    '''choose the number of shots or default to zero-shot prompting'''

    if not args.method == "zero-shot":
        MapToShot = {
                    "two-shot": 2, "three-shot": 3, "four-shot": 4,
                    "five-shot": 5, "six-shot": 6, "seven-shot": 7,
                    "eight-shot": 8, "nine-shot": 9, "eight-shot": 10, "twelve-shot": 12
                }    
        test_dataset = InstructionDataset(
            test_data, tokenizer, is_train=False,
            incontext=exemplars, max_length=4096, num=MapToShot[args.method], input_type=args.input_type
            )
    else:  # zero-shot
        del exemplars # free up memory
        test_dataset = InstructionDataset(
            test_data, tokenizer, is_train=False, max_length=512,
                incontext=False, input_type=args.input_type
            )
    return test_dataset    
    
    

    
    
    
    
def generate_data(data_path: str = None, 
                  valid_path: str = None, 
                  inshot_data_path: str = None,      
                  valid_size: float = 0.15, 
                  train_val: bool = True, 
                  data_format: str = 'json'):
    '''Load data & return splits'''
    
    if train_val:
        # Loading dataset for training and validation
        train = load_dataset(data_format, data_files=data_path)
        valid = load_dataset(data_format, data_files=valid_path)
        
        valid = valid['train'].train_test_split(
            test_size=valid_size, 
            shuffle=True, 
            seed=42
        )
        train_ds = train['train']  # Training data
        valid_ds = valid['test']   # Validation data
        return train_ds, valid_ds
    else:
        # Loading few-shot and query datasets
        fewshot = load_dataset(data_format, data_files=inshot_data_path, split='train')  # Few-shot data
        query = load_dataset(data_format, data_files=data_path, split='train')          # Query data
        return query, fewshot      
    
    
    
    
    
    
