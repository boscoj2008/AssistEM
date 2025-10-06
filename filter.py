#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
import json
import datasets
import argparse
import random
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer

# bulk import 
from utils import (
process_batch, 
tfidf_sim, 
calculate_information_density, 
clusterArray_to_blockDict, 
all_in_one_clusterize, 
proportional_class_distribution_sampling
)
from collections import Counter, defaultdict
    
import logging

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List



def timer(label: str, log=logging.info):
    """Lightweight timing context manager."""
    class _T:
        def __enter__(self):
            self.t0 = perf_counter()
            return self
        def __exit__(self, *exc):
            dt = perf_counter() - self.t0
            log(f"{label} in {dt:.2f}s")
    return _T()


def score(entry: Dict[str, Any], alpha: float) -> float:
    """Weighted score for a dataset entry."""
    return alpha * entry["tfidf_norm"] + (1.0 - alpha) * entry["ner_score"]


def domain_means(train_split: Iterable[Dict[str, Any]], alpha: float) -> Dict[str, float]:
    """Compute mean weighted score per domain."""
    totals = defaultdict(float)
    counts = Counter()
    for e in train_split:
        d = e["domain"]
        totals[d] += score(e, alpha)
        counts[d] += 1
    return {d: totals[d] / counts[d] for d in counts}


def filter_below_domain_mean(train_split: Iterable[Dict[str, Any]], means: Dict[str, float], alpha: float):
    """Keep entries whose score is below their domain's mean."""
    keep = []
    for e in train_split:
        d = e["domain"]
        if means[d] > score(e, alpha):
            keep.append(e)
    return keep



@dataclass(frozen=True)
class HP:
    dataset: Path
    cluster_method: str
    n_neighbors: int
    model_name: str
    max_seq_length: int
    alpha: float
    target_samples: int


def parse_args() -> HP:
    p = argparse.ArgumentParser(description="Data reduction")
    p.add_argument("--dataset", type=Path, default=Path("train.json"),
                   help="Path to JSON dataset (HuggingFace 'json' loader format).")
    p.add_argument("--community_algorithm", type=str, default="louvain",
                   help="Clustering algorithm (e.g., 'louvain' or 'leiden').")
    p.add_argument("--num_clusters", type=int, default=10,
                   help="Number of neighbors for the kNN graph (n_neighbors).")
    p.add_argument("--model_name", type=str, default="output/tsdae-model",
                   help="SentenceTransformer model path or name.")
    p.add_argument("--max_seq_length", type=int, default=256,
                   help="Model sequence length.")
    p.add_argument("--alpha", type=float, default=0.9,
                   help="Weight for tfidf_norm; (1 - alpha) is weight for ner_score.")
    p.add_argument("--target_samples", type=int, default=4000,
                   help="Total samples to select after clustering & sampling.")

    args, _ = p.parse_known_args()

    return HP(
        dataset=args.dataset,
        cluster_method=args.community_algorithm,  # normalize name in code
        n_neighbors=args.num_clusters,           # clearer meaning than "num_clusters"
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        alpha=args.alpha,
        target_samples=args.target_samples,
    )




def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    hp = parse_args()

    logging.info("Hyperparams: %s", {
        "dataset": str(hp.dataset),
        "cluster_method": hp.cluster_method,
        "n_neighbors": hp.n_neighbors,
        "model_name": hp.model_name,
        "max_seq_length": hp.max_seq_length,
        "alpha": hp.alpha,
        "target_samples": hp.target_samples,
    })

    with timer("Loading dataset"):
        ds = datasets.load_dataset("json", data_files=str(hp.dataset))
        train = ds["train"]

    # Sanity check: required columns
    required_cols = {"instruction", "input", "output", "domain", "tfidf_norm", "ner_score"}
    missing = required_cols.difference(train.column_names)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

    # Domain means and filtering
    with timer("Computing per-domain means"):
        means = domain_means(train, hp.alpha)

    with timer("Filtering by below-domain-mean score"):
        filtered: List[Dict[str, Any]] = filter_below_domain_mean(train, means, hp.alpha)

    if not filtered:
        raise ValueError("No samples remained after filtering; check alpha or data quality.")

    # Build embeddings
    with timer("Loading SentenceTransformer model"):
        model = SentenceTransformer(hp.model_name)
        model.max_seq_length = hp.max_seq_length

    sentences = [row["input"] for row in filtered]
    with timer(f"Encoding {len(sentences)} sentences"):
        embeddings = model.encode(sentences, show_progress_bar=True)

    # Graph-cluster embeddings
    with timer(f"Clustering with '{hp.cluster_method}' (n_neighbors={hp.n_neighbors})"):
        clusters: Dict[int, List[int]] = all_in_one_clusterize(
            vectors=embeddings,
            method=hp.cluster_method,
            n_neighbors=hp.n_neighbors,
        )
        # clusters: {cluster_id -> [indices into 'embeddings' / 'filtered']}

    # Cluster sizes 
    cluster_sizes = {cid: len(members) for cid, members in clusters.items()}
    if not cluster_sizes:
        raise ValueError("No clusters were formed; check clustering settings/data.")

    # Proportional sampling across clusters
    with timer(f"Proportional sampling to {hp.target_samples}"):
        selected_indices = proportional_class_distribution_sampling(
            clusters, cluster_sizes, filtered, hp.target_samples
        )

    reduced = [filtered[i] for i in selected_indices]
    size = len(reduced)

    # Save
    alpha_tag = int(hp.alpha * 10)
    beta_tag = int((1.0 - hp.alpha) * 10)
    size_k = round(size / 1000.0, 1)
    out_path = Path(f"train_alpha_{alpha_tag}_beta_{beta_tag}_{size_k}k.json")

    with timer(f"Saving {size} samples to {out_path}"):
        out_path.write_text(json.dumps(reduced, indent=2), encoding="utf-8")

    logging.info("Dataset saved at: %s", out_path.resolve())


if __name__ == "__main__":
    main()





