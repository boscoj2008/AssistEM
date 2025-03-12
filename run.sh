
python3 finetune.py --base_model_path 'Deci/DeciLM-7B' --train_data_path 'dataset/itdf/reducted_trainset-4.0k.json' --batch_size 32  --accum_grad_step 1 --epoch 3  --lr 3e-4 --lr_scheduler 'cosine' --peft_type 'LoRA' --linear_layers --use_gradient_checkpointing  




python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-text-w/semi-text-w-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --incontext_data_path 'dataset/magellan/dirty/walmart-amazon/walmart-amazon-dirty-test.json' 


python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/magellan/dirty/amazon-google/amazon-google-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --incontext_data_path 'dataset/magellan/dirty/walmart-amazon/walmart-amazon-dirty-test.json' 

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/rel-text/rel-text-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --incontext_data_path 'dataset/magellan/dirty/walmart-amazon/walmart-amazon-dirty-test.json' 

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-text-c/semi-text-c-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --incontext_data_path 'dataset/magellan/dirty/walmart-amazon/walmart-amazon-dirty-test.json' 










