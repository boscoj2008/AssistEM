# Optimizing Sample Selection for Large Language Model-based Entity Matching using AssistEM

abstract 

The meteoric rise of large language models (LLMs) has reshaped natural language processing, inspiring new approaches to data integration tasks such as entity matching (EM). While proprietary models like GPT-4 deliver strong performance, open-source alternatives (e.g., Mistral, DeciLM, Gemma-3) require supervised fine-tuning (SFT) to excel in specialized domains. However, naïvely training billion-parameter scale LLMs on uncurated corpora is computationally prohibitive and environmentally costly, often taking days while introducing inefficiencies that limit rapid application deployment.

We introduce AssistEM, a framework for efficient LLM adaptation to EM via principled data selection. AssistEM integrates NER and TF-IDF signals into a composite score, filters entity pairs with domain-level thresholds, constructs denoised semantic embeddings, and applies graph-based clustering with class-aware sampling to obtain a compact yet representative set of 8,000 high-quality training instances. This yields rapid specialization: AssistEM-trained LLMs converge in under 4.5 hours—versus 12+ hours for DeciLM-7B and 24+ hours for Yi-34B—while surpassing GPT-4-0613 by 4.3 F1 points and Ditto by 11 F1 points across diverse EM benchmarks. 

By aligning data quality with model specialization, AssistEM demonstrates that selective fine-tuning not only accelerates adaptation but also improves training efficiency (requiring fewer GPU hours), enabling open-source LLMs to rival—and in some cases outperform—closed-source models. These results highlight data selection as a critical lever for sustainable, domain-specific LLM deployment.

Repository contains code for [Optimizing Sample Selection for Large Language Model-based Entity Matching using AssistEM](https://pakdd2025.org/call-for-paper-llm/) accepted in PAKDD conference 2025 and recently extended & submitted to the journal of data science and analytics.

![AssistEM pipeline. ](./img/method.png)

## Requirements

```
python install -r requirements.txt
```

## Datasets

We use eight real-world benchmark datasets with different structures from [Machamp](https://github.com/megagonlabs/machamp) , [Geo-ER](https://github.com/PasqualeTurin/Geo-ER) also used in the 
[PromptEM paper](https://arxiv.org/abs/2207.04802) and some DeepMatcher benchmarks.



## Training embedding model
```
python tsdae-train.py --dataset dataset/train-full.json --batch_size 32 --epochs 10
```


## NER model scorer
```
python ner_scorer.py --dataset dataset/train-full.json --ner_model 'en_core_web_md' 
```

## TF-IDF scorer
```
use tfidf_sim in utils for new datasets
```


## Filter dataset
- set desired alpha and sample size or use generate_samples.sh to iterate multiple sample-sizes
```
python filter.py --dataset train-full.json --alpha 0.9 --target_samples 8000' 
```


## Fine-tuning

```
python3 finetune.py --base_model_path 'Deci/DeciLM-7B' --train_data_path 'dataset_path' --batch_size 32  --accum_grad_step 1 --epoch 3  --lr 3e-4 --lr_scheduler 'cosine' --peft_type 'LoRA' --linear_layers --use_gradient_checkpointing  
```

## Evaluation
 
```
python3 evaluate.py --method 'zero-shot' --test_data_path 'dataset_path' --base_model_path 'Deci/DeciLM-7B' --peft_path 'checkpoint/LoRA_deci' --incontext_data_path 'dataset_path' 
```
