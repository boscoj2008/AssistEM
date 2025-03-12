import torch
import json
import time
import os
import numpy as np
#import community as community_louvain
#import networkx as nx
#from cdlib import algorithms
from transformers import BitsAndBytesConfig
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances
from torch.nn import functional as F




def entity_matching_prompt1(query, exmpls: dict, role: str = "Data Engineer", num: int = 3) -> str:
    char = '\n'
    if exmpls:
        in_context = random_sampling(exmpls, num=num)

        prompt = f"You're an AI assistant assisting a {role.lower()} with entity matching in the {in_context[0]['domain']} domain. Here are some examples to guide you:\n\n"
        for i, context in enumerate(in_context, start=1):
            prompt += f"{i}. {role}: {context['instruction']}\n {context['input'].lower()}\n   ASSISTANT: {context['output']}.\n\n"
        prompt += f"Now, consider this query: {query['instruction']}\n   {role}: {query['input'].lower()}\n   ASSISTANT:"
    else:
        prompt = f"You're an AI assistant specializing in entity matching, assisting a {role.lower()} in the {query['domain']} domain. {query['instruction']}\n{role}: {query['input'].lower()}\nASSISTANT:"
    
    return prompt
    
    
def entity_matching_prompt(query, exmpls: dict, role: str = "Data Engineer", num: int = 3, input_type: str = "BrandTitle" ) -> str:
    char = '\n'
    if exmpls:
        in_context = random_sampling(exmpls, num=num)
        
        prompt = f"You're an AI assistant assistiupdate_json_resultng a {role.lower()} with entity matching in the {in_context[0]['domain']} domain. Here are some examples to guide you:\n\n"
        for i, context in enumerate(in_context, start=1):
            if len(context['input']) > 1 and input_type != str(None):
               
                prompt += f"{i}. {role}: {context['instruction']} {context['input'][input_type].lower()}\n   ASSISTANT: {context['output']}.\n\n"
            else:    
                prompt += f"{i}. {role}: {context['instruction']} {context['input'].lower()}\n   ASSISTANT: {context['output']}.\n\n"
        if len(query['input']) > 1 and input_type != str(None):
            prompt += f"Now, consider this query: {query['instruction']}\n   {role}: {query['input'][input_type].lower()}\n   ASSISTANT:"
        else:
            prompt += f"Now, consider this query: {query['instruction']}\n   {role}: {query['input'].lower()}\n   ASSISTANT:"
    else:
        if input_type == str(None):
            
            prompt = f"You're an AI assistant specializing in entity matching, assisting a {role.lower()} in the {query['domain']} domain. {query['instruction']}\n{role}: {query['input'].lower()}\nASSISTANT:"
        else:
            prompt = f"You're an AI assistant specializing in entity matching. Perform an action that satisfies the following request.\nASSISTANT:"
    
    return prompt   



def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    
def random_sampling(sentences, num):
    """randomly sample subset of the training pairs"""
    idxs = np.random.choice(len(sentences), size=num, replace=False)
    selected_sentences = [sentences[int(i)] for i in idxs]
    return deepcopy(selected_sentences)    
        


def balanced_sampling(sentences, num):
    """Randomly sample a balanced subset of 'yes' and 'no' answers"""
    # Separate 'yes' and 'no' answers
    yes_answers = [s for s in sentences if s['answer'] == 'yes']
    no_answers = [s for s in sentences if s['answer'] == 'no']

    # Calculate the number of samples needed for 'yes' and 'no' to balance
    num_each = num // 2

    # Randomly select 'yes' and 'no' samples to create a balanced set
    selected_sentences = []
    if len(yes_answers) >= num_each and len(no_answers) >= num_each:
        selected_sentences.extend(np.random.choice(yes_answers, size=num_each, replace=False))
        selected_sentences.extend(np.random.choice(no_answers, size=num_each, replace=False))
    else:
        # If insufficient samples for balance, use all available samples
        selected_sentences.extend(yes_answers)
        selected_sentences.extend(no_answers)

    return deepcopy(selected_sentences)    
    
    
    
    
def update_json_result(file_name, evaluation_result):


    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
    else:
        data = {}
        
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data[timestamp] = evaluation_result
    print(data)
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)
        
        
def clusterArray_to_blockDict(clusters):
    blocks = {}
    for index, value in enumerate(clusters):
        if value in blocks.keys():
            blocks[value].append(index)
        else:
            blocks[value] = list()
            blocks[value].append(index)
    return blocks



def vec_2_graph(vectors, N):
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph
    # build the graph which is full-connected
    mat = kneighbors_graph(vectors, N, metric='cosine', mode='distance', include_self=True, n_jobs=-1)
    mat.data = 1 - mat.data  # to similarity
    g = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())
    print(nx.info(g))
    return g 


def cluster_graph(vectors, clusteriser="louvian", N=10 ):
    """"clusterise using louvian, leiden, markov, or eig """
    import pandas as pd
    
    if clusteriser == 'louvian':
        print("using louvian clusterizer")
        start = time.time()
        graph = vec_2_graph(vectors, N) # init graph
        end = time.time()
        dif = end - start        
        print("graph constructed in {:.2f} seconds".format(dif))
        
        start_blocking = time.time()
        partition = community_louvain.best_partition(graph, random_state=42)
        end_blocking = time.time()
        dif = end_blocking - start_blocking
        print("blocking completed in {:.2f} seconds".format(dif))
        
        return clusterArray_to_blockDict(partition.values())
    
    if clusteriser == 'leiden':
        print("using leiden clusterizer")
        start = time.time()
        graph = vec_2_graph(vectors, N) # init graph
        end = time.time()
        dif = end - start
        print("graph constructed in {:.2f} seconds".format(dif))
        
        start_blocking = time.time()
        comms = algorithms.leiden(graph)
        comms = comms.communities
        end_blocking = time.time()
        dif = end_blocking - start_blocking
        print("blocking completed in:", dif,' seconds')
        comms = dict(sorted({item:list_id for list_id, l in enumerate(comms) for item in l}.items()))
        return clusterArray_to_blockDict(pd.Series(comms).values) 
    
    


def NEFTune(model, noise_alpha=5):
    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha/torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            else:
                return orig_embed(x)
        return new_func
    ##### NOTE: this is for a LLaMA model ##### 
    ##### For a different model, you need to change the attribute path to the embedding #####
    orig_forward = model.base_model.embed_tokens.forward
    model.base_model.embed_tokens.forward = noised_embed(orig_forward, noise_alpha)
    return model    
    
