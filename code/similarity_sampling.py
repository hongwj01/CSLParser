import numpy as np
import pandas as pd
from collections import Counter
import heapq
import time
import calendar
import json
from copy import deepcopy
import re
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import os
from tqdm import tqdm
from config import datasets, benchmark


def get_embeddings_tfidf(logs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(logs)
    svd = TruncatedSVD(n_components=min(
        512, tfidf_matrix.shape[1]), random_state=42)
    reduced_embeddings = svd.fit_transform(tfidf_matrix)
    embeddings = [reduced_embeddings[i]
                  for i in range(reduced_embeddings.shape[0])]
    return embeddings


def select_representative_logs(embeddings, k=10):
    similarity_matrix = cosine_similarity(embeddings)
    avg_similarities = similarity_matrix.mean(axis=1)
    representative_indices = np.argsort(avg_similarities)[-k:]
    return representative_indices


def calculate_cross_partition_similarity(logs_embedding, selected_embeddings):
    if len(selected_embeddings) == 0:
        return np.zeros((len(logs_embedding), 1))
    return cosine_similarity(logs_embedding, selected_embeddings)


def select_different_log(cross_partition_similarities, k=5):
    avg_similarities = cross_partition_similarities.mean(axis=1)
    different_indices = np.argsort(avg_similarities)[:k]
    return different_indices


class Vocab:
    def __init__(self, stopwords=["<*>"]):
        stopwords = ["a", "an", "and", "i", "ie", "so", "to", "the",] \
            + list(calendar.day_name) + list(calendar.day_abbr) \
            + list(calendar.month_name) + list(calendar.month_abbr)
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))

    def build(self, sequences):
        print("Build vocab with examples: ", len(sequences))
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = self.__filter_stopwords(sequence)
        token_count = [(token, self.token_counter[token])
                       for token in set(sequence)]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: x[1])
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]


def clean(s):
    s = s.lower().strip()
    log_format = re.sub(r'[0-9A-Za-z, ]+', '', s)
    unique_chars = list(set(log_format))
    sorted_string = ''.join(sorted(unique_chars))
    s = re.sub('\+|\_|\#|:|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.?!', ' ', s)
    s = " ".join([word for word in s.strip().split()
                 if not bool(re.search(r'\d', word))])
    return s, sorted_string


def hierarchical_clustering(contents):
    vocab = Vocab()
    vocab.build([v[0].split() for v in contents.values()])

    hierarchical_clusters = {}
    for k, v in contents.items():
        tokens = v[0].split()
        token_length = len(tokens)
        frequent_token = tuple(sorted(vocab.topk_tokens(tokens, 3)))
        log_format = v[1]
        cluster_key = (token_length, log_format)

        if frequent_token not in hierarchical_clusters:
            hierarchical_clusters[frequent_token] = {
                "size": 1, "cluster": {cluster_key: [k]}}
        else:
            hierarchical_clusters[frequent_token]["size"] = hierarchical_clusters[frequent_token]["size"] + 1
            if cluster_key not in hierarchical_clusters[frequent_token]["cluster"]:
                hierarchical_clusters[frequent_token]["cluster"][cluster_key] = [
                    k]
            else:
                hierarchical_clusters[frequent_token]["cluster"][cluster_key].append(
                    k)

    print("Number of coarse-grained clusters: ",
          len(hierarchical_clusters.keys()))
    total_fine_clusters = 0
    for k, v in hierarchical_clusters.items():
        total_fine_clusters += len(hierarchical_clusters[k]["cluster"])
    print("Number of fine-grained clusters: ", total_fine_clusters)
    return hierarchical_clusters


def hierarchical_distribute(hierarchical_clusters, shot, logs, embeddings):
    candidate_samples = []
    coarse_clusters = hierarchical_clusters.keys()
    coarse_clusters = sorted(
        coarse_clusters, key=lambda x: hierarchical_clusters[x]["size"], reverse=True)
    corase_size = len(coarse_clusters)
    empty_clusters = []
    selected_embeddings = []
    print("Shot: ", shot, "Coarse size: ", corase_size)

    while shot > 0:
        for coarse_id, coarse_key in enumerate(coarse_clusters):
            if coarse_key in empty_clusters:
                continue
            coarse_quota = max(int(shot // corase_size), 1)

            fine_clusters = hierarchical_clusters[coarse_key]["cluster"].keys()
            fine_clusters = sorted(fine_clusters, key=lambda x: len(
                hierarchical_clusters[coarse_key]["cluster"][x]), reverse=True)
            fine_size = len(fine_clusters)
            cluster_size = 0
            for fine_id, fine_key in enumerate(fine_clusters):
                fine_quota = min(
                    shot, int(coarse_quota // fine_size) + (fine_id < coarse_quota % fine_size))
                fine_quota = min(fine_quota, len(
                    hierarchical_clusters[coarse_key]["cluster"][fine_key]))
                if fine_quota == 0:
                    continue

                cluster_ids = hierarchical_clusters[coarse_key]["cluster"][fine_key]
                cluster_embeddings = [embeddings[i] for i in cluster_ids]

                num_indices = min(fine_quota * 10000, len(cluster_ids))
                ids = np.random.choice(
                    len(cluster_ids), num_indices, replace=False)
                random_ids = [cluster_ids[i] for i in ids]
                random_samples_embeddings = [
                    cluster_embeddings[i] for i in ids]

                representative_ids = select_representative_logs(
                    random_samples_embeddings, min(fine_quota * 10, len(cluster_ids)))
                representative_samples = [random_ids[i]
                                          for i in representative_ids]
                representative_samples_embeddings = [
                    random_samples_embeddings[i] for i in representative_ids]

                cross_similarity = calculate_cross_partition_similarity(
                    representative_samples_embeddings, selected_embeddings)
                different_ids = select_different_log(
                    cross_similarity, fine_quota)

                samples = [representative_samples[i] for i in different_ids]
                samples_embeddings = [
                    representative_samples_embeddings[i] for i in different_ids]

                candidate_samples.extend(samples)
                selected_embeddings.extend(samples_embeddings)
                shot -= fine_quota

                sample_ids = []
                for i in range(len(cluster_ids)):
                    for sample in samples:
                        if sample == cluster_ids[i]:
                            sample_ids.append(i)
                            break

                for i in sorted(sample_ids, reverse=True):
                    cluster_ids.pop(i)
                cluster_size += len(
                    hierarchical_clusters[coarse_key]["cluster"][fine_key])
            if cluster_size == 0:
                empty_clusters.append(coarse_key)
                corase_size -= 1

    return candidate_samples


def sampling_random(logs, labels=None, shots=[32]):
    logs, labels = zip(*list(set(zip(logs, labels))))
    contents = {}
    for i, x in enumerate(logs):
        x, fx = clean(x)
        if len(x.split()) > 0:
            contents[i] = (x, fx)

    sample_candidates = {}
    for idx, shot in enumerate(shots):
        begin_time = time.time()
        sampled_ids = random.sample(list(contents.keys()), shot)
        samples = [(logs[i], labels[i]) for i in sampled_ids]
        sample_candidates[shot] = samples
        end_time = time.time()
        print(f"{shot}-shot random sampling time: ", (end_time - begin_time))

    return sample_candidates


def sampling_tfidf(logs, labels=None, shots=[32], data_dir='../datasets/loghub-2.0-full', dataset='Apache'):
    logs, labels = zip(*list(set(zip(logs, labels))))
    contents = {}
    for i, x in enumerate(logs):
        x, fx = clean(x)
        if len(x.split()) > 0:
            contents[i] = (x, fx)

    begin_time = time.time()
    hierarchical_clusters = hierarchical_clustering(contents)
    end_time = time.time()
    clustering_time = end_time - begin_time
    print("hierarchical clustering time: ", clustering_time)

    embeddings_path = f'{data_dir}/{dataset}/embedding_tfidf.pickle'
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = get_embeddings_tfidf(logs)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)

    sample_candidates = {}
    for idx, shot in enumerate(shots):
        begin_time = time.time()
        sampled_ids = hierarchical_distribute(
            deepcopy(hierarchical_clusters), shot, logs, embeddings)
        if labels is not None:
            samples = [(logs[i], labels[i]) for i in sampled_ids]
        else:
            samples = [(logs[i], logs[i]) for i in sampled_ids]
        sample_candidates[shot] = samples
        end_time = time.time()
        print(f"{shot}-shot sampling time: ", (end_time - begin_time))

    return sample_candidates
