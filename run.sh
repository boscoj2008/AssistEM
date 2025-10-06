


#python3 finetune.py --base_model_path '01-ai/Yi-34B' --train_data_path 'train_alpha_9.0_beta_1.0_8.0k.json' --batch_size 32  --accum_grad_step 1 --epoch 3  --lr 3e-4 --lr_scheduler 'cosine' --peft_type 'QLoRA' --linear_layers --use_gradient_checkpointing

#python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/wdc/wdc-test.json' --base_model_path '01-ai/Yi-34B' --peft_path 'checkpoint/QLoRA_llama' --train_set_size 4000 --input_type None

#python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-heter/semi-heter-test.json' --base_model_path '01-ai/Yi-34B' --peft_path 'checkpoint/QLoRA_llama' --train_set_size 4000

#python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/magellan/dirty/amazon-google/amazon-google-test.json' --base_model_path '01-ai/Yi-34B' --peft_path 'checkpoint/QLoRA_llama' --train_set_size 4000

#python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/rel-text/rel-text-test.json' --base_model_path '01-ai/Yi-34B' --peft_path 'checkpoint/QLoRA_llama' --train_set_size 4000

#python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-homo/semi-homo-test.json' --base_model_path '01-ai/Yi-34B' --peft_path 'checkpoint/QLoRA_llama' --train_set_size 4000

#python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/geo-heter/geo-heter-test.json' --base_model_path '01-ai/Yi-34B' --peft_path 'checkpoint/QLoRA_llama' --train_set_size 4000

#python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-rel/semi-rel-test.json' --base_model_path '01-ai/Yi-34B' --peft_path 'checkpoint/QLoRA_llama' --train_set_size 8000









python3 finetune.py --base_model_path 'Deci/DeciLM-7B' --train_data_path 'train_alpha_9.0_beta_1.0_8.0k.json' --batch_size 32  --accum_grad_step 1 --epoch 3  --lr 3e-4 --lr_scheduler 'cosine' --peft_type 'LoRA' --linear_layers --use_gradient_checkpointing


python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/wdc/wdc-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --train_set_size 4000 --input_type None

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-heter/semi-heter-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --train_set_size 4000

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/magellan/dirty/amazon-google/amazon-google-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --train_set_size 4000

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/rel-text/rel-text-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --train_set_size 4000

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-homo/semi-homo-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --train_set_size 4000

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/geo-heter/geo-heter-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --train_set_size 4000

python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset/machamp/semi-rel/semi-rel-test.json' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --train_set_size 4000


































