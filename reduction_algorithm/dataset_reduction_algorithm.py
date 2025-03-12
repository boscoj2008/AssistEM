#!/usr/bin/env python
# coding: utf-8


import json
import datasets
import argparse
import random
import numpy as np
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
# import spacy
from tqdm import tqdm
from thinc.api import require_gpu  
from thinc.backends import use_pytorch_for_gpu_memory 

from joblib import Parallel, delayed
# import community as community_louvain
import community.community_louvain
import networkx as nx
from cdlib import algorithms
import pandas as pd
import time 
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances
import numpy as np


use_pytorch_for_gpu_memory()
require_gpu()
# nlp = spacy.load("en_core_web_md")



# convert labels 
def process_batch(batch):
    return {'label_string': ["yes" if item == 1 else "no" for item in batch['label']]}



def random_sampling(sentences, num):
    """randomly sample subset of the training pairs"""
    np.random.seed(seed=42)
    idxs = np.random.choice(len(sentences), size=num, replace=False)
    selected_sentences = [sentences[int(i)] for i in idxs]
    return deepcopy(selected_sentences)  


def tfidf_sim(left_tupple, right_tuple):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([left_tupple, right_tuple])
    average_infor_density = vectors.mean()
    return average_infor_density


 
def calculate_information_density(text1, text2):

    # Process text1
    doc1 = nlp(text1)
    entities1 = [(ent.text, ent.label_) for ent in doc1.ents]
    information_density1 = len(entities1) / len(doc1)

    # Process text2
    doc2 = nlp(text2)
    entities2 = [(ent.text, ent.label_) for ent in doc2.ents]
    information_density2 = len(entities2) / len(doc2)

    # Calculate the average information density
    average_information_density = (information_density1 + information_density2) / 2

    return average_information_density

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
    return g 


def all_in_one_clusteriser(vectors, clusteriser="louvian", N=10 ):
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
        partition = community.community_louvain.best_partition(graph, random_state=42)
        end_blocking = time.time()
        dif = end_blocking - start_blocking
        print("clustering completed in {:.2f} seconds".format(dif))
        
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
        print("clustering completed in:", dif,' seconds')
        comms = dict(sorted({item:list_id for list_id, l in enumerate(comms) for item in l}.items()))
        return clusterArray_to_blockDict(pd.Series(comms).values) 
    

    
def process_tuple(instruct_prompt, input_text, output, domain):
    try:
        parts = input_text.split('entity')
        if len(parts) > 2:
            left_ = parts[1].strip()[2:]
            right_ = parts[2].strip()[2:]
        else:
            parts = input_text.split('product')
            left_ = parts[1].strip()[2:]
            right_ = parts[2].strip()[2:]
        score_ = tfidf_sim(left_, right_)
        return {
            'instruction': instruct_prompt,
            'input': f'product 1: {left_.lower()} product 2: {right_.lower()}',
            'output': f"The answer is {output}",
            'domain': domain,
            'info_density': score_
        }
    except IndexError:
        return None
    



parser = argparse.ArgumentParser(description='data reduction')
parser.add_argument("--dataset", type=str,default='train-full.json', help='dataset name i.e json format')
parser.add_argument("--community_algorithm", type=str,default='louvian', help='clustering algorithm')
parser.add_argument("--num_clusters", type=int, default=10, help='used in knn graph')
parser.add_argument("--instruction", 
                    default="Analyze the product descriptions and determine if they are describing the same product. Respond with yes if they do and no if they don't")
parser.add_argument("--model_name", type=str, default='all-MiniLM-L6-v2',help="model_path or name")
parser.add_argument("--max_seq_length", type=int, default=256, help='models sequence length')

hp, _ = parser.parse_known_args()

key_values = {
    'dataset': hp.dataset,
    'cluster_method': hp.community_algorithm,
    'num_clusters': hp.num_clusters,
    'model_name': hp.model_name,
}




dataset = datasets.load_dataset('json', data_files=hp.dataset)
instructions = dataset['train']['instruction']
inputs = dataset['train']['input']
outputs = dataset['train']['output']
domains = dataset['train']['domain']






results = Parallel(n_jobs=-1)(
    delayed(process_tuple)(inp, inst, output, domain)
    for inp, inst, output, domain in zip(instructions, inputs, outputs, domains)
)

train_formatted = [r for r in results if r is not None]


info_density_values = [x['info_density'] for x in train_formatted]



mean_info_density = np.mean(info_density_values)

tmp = []

for instance in train_formatted:
    if instance['info_density'] > mean_info_density:
        tmp.append(instance)



model = SentenceTransformer(hp.model_name)

sentences = [s['input'] for s in tmp]

embeddings = model.encode(sentences, show_progress_bar=True)


data = all_in_one_clusteriser(embeddings, clusteriser=hp.community_algorithm, N= hp.num_clusters)



cluster_sts = {}
for cluster_id, cluster_members in data.items():
    cluster_sts[cluster_id] = len(cluster_members)




plt.figure(figsize=(15,10))
plt.title('Enity Coverage Distribution')
plt.bar(cluster_sts.keys(), cluster_sts.values())
plt.hlines(50, xmin=-1, xmax=80, colors='red', alpha=0.6)
plt.xlabel('Cluster ID')
plt.ylabel('Cluster Entity Count')
plt.show() 


# 49 -> 4023
# 81 -> 6226
#  114 -> 8155
# 155 -> 10227
# 220 -> 12628
#  281 -> 14182
#  405 -> 15818




smp_thresh = [49, 81, 114, 155, 220, 281, 405]

for threshold in smp_thresh:
    train = []
    combined_ids = []
    for cluster_idx, total_clus_mem in cluster_sts.items():
    # min/max handle
        cluster_members = data[cluster_idx]
        if total_clus_mem < threshold:
            num_samples = total_clus_mem
        elif total_clus_mem > threshold:
            num_samples = threshold
            
        sample_ids = random.sample(cluster_members, k=num_samples)
        combined_ids.extend(sample_ids)

    

    random.shuffle(combined_ids)
    train = [tmp[idx] for idx in combined_ids]
    size = len(train)

    with open(f'train_tfidf_{round(size/1000, 1)}k.json','w') as f:
        json.dump(train, f, indent=2)







