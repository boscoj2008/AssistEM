from __future__ import annotations


import torch
import json
import time
import os
import numpy as np
import faiss
from transformers import BitsAndBytesConfig
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances
from torch.nn import functional as F

from contextlib import contextmanager
from time import perf_counter

from typing import Dict, Any, List, Iterable
import random
import community.community_louvain as community_louvain
import networkx as nx
from cdlib import algorithms
from collections import Counter, defaultdict
from tqdm import tqdm



def entity_matching_prompt(
    query: Dict[str, Any],
    role: str = "Data Engineer",
    num: int = 3,
    input_field: str = "BrandTitle",
) -> str:
    """
    Build a natural-language prompt for an entity-matching AI assistant.

    Parameters
    ----------
    query : dict
        Query example containing 'instruction' and 'input'.
    role : str, optional
        The role of the user, e.g. "Data Engineer".
    num : int, optional
        Number of in-context examples to include.
    input_field : str or None, optional
        Field in the input dict to use, e.g. "BrandTitle".

    Returns
    -------
    str
        The constructed prompt string.
    """

    lines = []

    instruction = query.get("instruction", "").strip()
    q_input = query.get("input")
    if isinstance(q_input, dict) and input_field in q_input:
        input_text = q_input[input_field].lower()
    else:
        input_text = str(q_input).lower()

        lines.append(
            f"You're an AI assistant specializing in entity matching. "
            f"Perform an action that satisfies the following request:\n"
            f"{instruction}\n{input_text}\nASSISTANT:"
        )

    return "\n".join(lines)


@contextmanager
def _timer(label: str, log: callable = print):
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        log(f"{label} in {dt:.2f} seconds")


def all_in_one_clusterize(
    vectors: Iterable[Any],
    method: str = "louvain",
    n_neighbors: int = 10,
    random_state: int = 42,
    log: callable = print,  # swap for `logging.info` in real apps
):
    """
    Cluster a k-NN graph built from `vectors` using the given `method`.

    Parameters
    ----------
    vectors : iterable-like
        Embeddings/feature vectors.
    method : {'louvain', 'leiden'}
        Community detection algorithm to use.
    n_neighbors : int
        Number of neighbors for the FAISS graph.
    random_state : int
        Seed/seed-equivalent for reproducibility.
    log : callable
        Logging sink (defaults to print).

    Returns
    -------
    dict
        returns for the label array.
    """
    

    method = method.lower()
    if method not in {"louvain", "leiden"}:
        raise ValueError("`method` must be 'louvain' or 'leiden'.")

    # 1) Build graph once (DRY) and time it
    with _timer("Graph constructed", log):
        graph = vec_2_graph_faiss(vectors, n_neighbors)

    # 2) Run clustering
    if method == "louvain":
        log("Using Louvain clustering")
        with _timer("Clustering completed", log):
            partition = community_louvain.best_partition(graph, random_state=random_state)
        # IMPORTANT: keep node order stable; don't rely on dict.values()
        labels = [partition[node] for node in graph.nodes()]

    else:  # method == "leiden"
        log("Using Leiden clustering")
        with _timer("Clustering completed", log):
            result = algorithms.leiden(graph, seed=random_state)
        # Build node -> label mapping, then align to graph node order
        node_to_label = {}
        for label, members in enumerate(result.communities):
            for node in members:
                node_to_label[node] = label
        labels = [node_to_label[node] for node in graph.nodes()]

    return clusterArray_to_blockDict(labels)



def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
       
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
        

# convert labels 
def process_batch(batch):
    return {'label_string': ["yes" if item == 1 else "no" for item in batch['label']]}


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




def vec_2_graph_faiss(vectors, N, metric='cosine', use_gpu=False):
    
    vectors = np.array(vectors, dtype=np.float32)
 
    n, d = vectors.shape
    
    if metric == 'cosine':
        norms = np.sqrt((vectors ** 2).sum(axis=1))
        vectors = vectors / norms[:, np.newaxis]
        index = faiss.IndexFlatIP(d)  
    else:
        index = faiss.IndexFlatL2(d)  
    

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    

    index.add(vectors)
    distances, neighbors = index.search(vectors, N)
    
    g = nx.Graph()
    g.add_nodes_from(range(n))
    
    for i in range(n):
        for j, d in zip(neighbors[i], distances[i]):
            if i != j: 
                weight = d if metric == 'cosine' else 1 - d  
                g.add_edge(i, j, weight=weight)
    
    return g



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
    

   
def proportional_class_distribution_sampling(data, cluster_sts, tmp, target_samples, random_seed=42):
    random.seed(random_seed)  # Ensure reproducibility
    
    # get global class distribution
    all_indices = sum(data.values(), [])  # All tuple indices across clusters
    class_counts = Counter(tmp[idx]['answer'] for idx in all_indices)
    total_tuples = sum(class_counts.values())
    class_proportions = {label: count / total_tuples for label, count in class_counts.items()}
    
    
    print(f"INFO: Global class proportions: {class_proportions}")
    
    
    target_per_class = {label: round(prop * target_samples) for label, prop in class_proportions.items()}
    print(f"INFO: Target samples per class: {target_per_class}")
    
    # Group indices by cluster and class
    class_indices = defaultdict(lambda: defaultdict(list))
    for cluster_idx, cluster_members in data.items():
        for idx in cluster_members:
            label = tmp[idx]['answer']  # Adjust key if needed
            class_indices[cluster_idx][label].append(idx)
    
    combined_ids = []
    total_items = sum(cluster_sts.values())
    
    # Initial sampling proportional to cluster size
    for cluster_idx, cluster_size in cluster_sts.items():
        # Proportion of samples for this cluster
        cluster_target = round((cluster_size / total_items) * target_samples)
        
        # Available indices per class in this cluster
        cluster_class_indices = class_indices[cluster_idx]
        
        # Allocate samples to match global class proportions
        cluster_samples = {}
        for label, prop in class_proportions.items():
            target_for_label = round(prop * cluster_target)
            available_indices = cluster_class_indices[label]
            num_samples = min(len(available_indices), target_for_label)
            if num_samples > 0:
                cluster_samples[label] = random.sample(available_indices, k=num_samples)
        
        # Combine samples from this cluster
        for label, indices in cluster_samples.items():
            combined_ids.extend(indices)
    
    # Adjust to match target_samples and class proportions
    if len(combined_ids) > 0:
        # Current class distribution
        current_dist = Counter(tmp[idx]['answer'] for idx in combined_ids)
        samples_by_class = {label: [idx for idx in combined_ids if tmp[idx]['answer'] == label]
                           for label in class_proportions}
        
        # Select samples to match target_per_class
        final_ids = []
        for label, target in target_per_class.items():
            available = samples_by_class.get(label, [])
            num_samples = min(len(available), target)
            if num_samples > 0:
                final_ids.extend(random.sample(available, k=num_samples))
        
        # fill remaining samples if needed
        remaining = target_samples - len(final_ids)
        if remaining > 0:
            all_available = sum([samples_by_class[label] for label in class_proportions], [])
            final_ids.extend(random.sample(all_available, k=min(remaining, len(all_available))))
        
        combined_ids = final_ids
    
    # Ensure exact target_samples
    random.shuffle(combined_ids)
    if len(combined_ids) > target_samples:
        combined_ids = random.sample(combined_ids, k=target_samples)
    elif len(combined_ids) < target_samples:
        print(f"Warning: Only {len(combined_ids)} samples collected, less than target {target_samples}")
    
    return combined_ids    
    
