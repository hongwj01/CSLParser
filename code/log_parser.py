import logging
import sys
import time
import os
import json
import pickle
import pandas as pd
import numpy as np
import string
import regex as re
import Levenshtein
from collections import Counter
from tqdm import tqdm
import torch
from transformers import set_seed
from arguments import get_args
from models import PreTrainedModel
from data_loader import DataLoaderForPromptTuning
from trainer import TrainingArguments, Trainer
from template_cache import LogTemplateCache
from llm_query import LLMForLogParsing
from similarity_sampling import *
from config import datasets, benchmark
from post_process import correct_single_template

logger = logging.getLogger("Log Parser")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger.handlers[0].setFormatter(formatter)


def sampling(data_args, common_args):
    log_file = benchmark[data_args.dataset_name]['log_file']
    labelled_logs = pd.read_csv(
        f'{data_args.data_dir}/{log_file}_structured.csv')
    k_rate = 0.2
    length = int(k_rate * len(labelled_logs))
    labelled_logs = labelled_logs[:length]
    raw_logs = labelled_logs['Content'].tolist()
    labels = labelled_logs['EventTemplate'].tolist()
    shots = [common_args.shot]

    sample_candidates = sampling_tfidf(
        raw_logs, labels, shots, data_args.data_dir, data_args.dataset_name)

    for shot, samples in sample_candidates.items():
        with open(f'{data_args.data_dir}/{data_args.dataset_name}/sampled_sim_{shot}.json', 'w') as f:
            for sample in samples:
                f.write(json.dumps(
                    {'log': sample[0], 'template': sample[1]}) + '\n')


def initialize_and_load_data(data_args, model_args):
    pretrained_model = PreTrainedModel(model_args.model_name_or_path)
    data_loader = DataLoaderForPromptTuning(data_args)
    logger.debug(
        f"{data_args.dataset_name} loaded with {len(data_loader.raw_datasets['train'])} train samples")

    pretrained_model.tokenizer = data_loader.initialize(
        pretrained_model.tokenizer)

    data_loader.tokenize()
    logger.debug(data_loader.processed_raw_datasets)

    data_loader.build_dataloaders(
        train_args.per_device_train_batch_size, train_args.per_device_eval_batch_size)
    logger.debug(f"Train dataloader: {data_loader.get_train_dataloader()}")

    return pretrained_model, data_loader


def setup_training(pretrained_model, data_loader, train_args, common_args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    devices = [device]

    training_args = TrainingArguments(
        output_dir=common_args.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.0,
        num_train_epochs=train_args.num_train_epochs,
        max_train_steps=train_args.max_train_steps,
        gradient_accumulation_steps=1,
        lr_scheduler_type="polynomial"
    )

    trainer = Trainer(
        model=pretrained_model,
        args=training_args,
        train_loader=data_loader.get_train_dataloader(),
        eval_loader=data_loader.get_val_dataloader(),
        compute_metrics=None,
        no_train_samples=len(data_loader.raw_datasets['train']),
        device=device,
    )

    return trainer, device, devices


def train_model(trainer, common_args, data_args):
    t0 = time.time()
    if common_args.load_model:
        pretrained_model = trainer.load_pretrained(
            f"{common_args.output_dir}/models/{data_args.dataset_name}")
    else:
        pretrained_model = trainer.train()
        if common_args.save_model:
            trainer.save_pretrained(
                f"{common_args.output_dir}/models/{data_args.dataset_name}")
    t1 = time.time()
    training_time = t1 - t0
    return pretrained_model, training_time


def split_by_delimiters(text):
    delimiters = [' ', '(', ')', '[', ']', '{', '}', ',', ';', '=']
    result = []
    start = 0
    for i, char in enumerate(text):
        if char in delimiters:
            if start < i:
                result.append(text[start:i])
            result.append(char)
            start = i + 1
    if start < len(text):
        result.append(text[start:])
    return result


def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def align_template(log, template):
    delimiters = [' ', '(', ')', '[', ']', '{', '}', ',', ';', '=']
    log_tokens = split_by_delimiters(log)
    template_tokens = split_by_delimiters(template)

    if template.count('<*>') == len(log_tokens) - len(template_tokens) + template.count('<*>'):
        return template

    corrected_template = []
    log_idx = 0
    template_idx = 0

    while log_idx < len(log_tokens) and template_idx < len(template_tokens):
        if template_tokens[template_idx] == '<*>':
            corrected_template.append('<*>')
            log_idx += 1
            template_idx += 1
        else:
            if log_tokens[log_idx] == template_tokens[template_idx]:
                corrected_template.append(template_tokens[template_idx])
                log_idx += 1
                template_idx += 1
            else:
                if log_tokens[log_idx] not in delimiters:
                    corrected_template.append('<*>')
                else:
                    corrected_template.append(log_tokens[log_idx])
                log_idx += 1

    while log_idx < len(log_tokens):
        if log_tokens[log_idx] not in delimiters:
            corrected_template.append('<*>')
        else:
            corrected_template.append(log_tokens[log_idx])
        log_idx += 1

    return ''.join(corrected_template)


def select_examples(log, examples, k=3):
    distances = []
    log_token = split_by_delimiters(log)
    for example in examples:
        example_token = split_by_delimiters(example['log'])
        distance = (-1) * lcs_length(log_token, example_token)
        distances.append((example, distance))

    distances.sort(key=lambda x: x[1])
    top_k_examples = []
    count = 0
    for example, _ in distances:
        example['template'] = align_template(
            example['log'], example['template'])
        top_k_examples.append(example)
        count += 1
        if count == k:
            break
    return top_k_examples


def check_template_is_balanced(template):
    brackets = {"(": ")", "[": "]", "{": "}", "<": ">"}
    quotes = {'"': '"'}

    stack = []

    for char in template:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets.get(stack.pop()) != char:
                return False
        elif char in quotes:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)

    return not stack


def check_log_and_template(log, template, confidences):
    confidence_threshold = 0.7
    keywords = [
        "error:", "failed:", "exception:",
        "crash:", "abort:", "fatal:", "warning:",
        "error=", "failed=", "exception=",
        "crash=", "abort=", "fatal=", "warning="
    ]

    count = template.count('<*>')
    for keyword in keywords:
        if keyword in log.lower() and count > 0:
            return "Keyword"

    template_tokens = split_by_delimiters(template)
    template_tokens = [
        token for token in template_tokens if token not in ['', ' ', '<*>']]
    if len(template_tokens) < 2 and count > 0:
        return "Keyword"

    tmp_template = template.replace("<*>", "").replace(" ", "")
    if not any(char not in string.punctuation for char in tmp_template):
        return "Keyword"

    if not check_template_is_balanced(template):
        return "Complex"

    if len(confidences) > 0:
        confidence = sum(confidences) / len(confidences)
    else:
        confidence = 1.0
    if confidence < confidence_threshold:
        return "Confidence"

    return "No"


def check_template_match(template, llm_template):
    constants = re.findall(r'[a-zA-Z0-9]+', template)
    constants = [c for c in constants if len(c) > 1]

    start_index = 0

    for constant in constants:
        index = llm_template.find(constant, start_index)
        if index == -1:
            return False
        start_index = index + len(constant)

    return True


def check_llm_template(log, template):
    tmp_template = template.replace("<*>", "")
    tmp_template = tmp_template.replace(" ", "")
    delimiters = [' ', '(', ')', '[', ']', '{', '}', ',', ';']
    if not any(char not in delimiters for char in tmp_template):
        return False

    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = ".*?".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"

    if log.count('|') > 15:
        return False

    try:
        matches = re.match(regex, log)
        return matches is not None
    except re.error:
        return False


def post_process_template(template):
    template = template.replace('<*)>', '<*>)')
    template = template.replace('<>', '<*>')
    template = template.replace('<$>', '<*>')
    template = template.replace('<*.>', '<*>.')
    template = template.replace('(<*)', '(<*>)')
    return correct_single_template(template)


def log_parse(data_args, model_args, common_args, pretrained_model, devices, vtoken="virtual-param"):
    logger.info("Starting template extraction")
    pretrained_model.eval()

    log_df = pd.read_csv(data_args.log_file)
    log_lines = log_df['Content'].tolist()
    with open(f'{data_args.data_dir}/{data_args.dataset_name}/sampled_sim_{common_args.shot}.json', 'rb') as f:
        examples = []
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    logger.info(f"Examples num: {len(examples)}")

    t0 = time.time()

    templates = []
    cache = LogTemplateCache()
    llm = LLMForLogParsing()
    pretrained_model_time, LLM_time, LLM_merge_time = 0, 0, 0
    LLM_tokens, LLM_merge_tokens = 0, 0
    LLM_invoke, LLM_merge_invoke = 0, 0
    LLM_complex_invoke, LLM_complex_tokens, LLM_complex_time = 0, 0, 0
    LLM_keyword_invoke, LLM_keyword_tokens, LLM_keyword_time = 0, 0, 0
    LLM_confidence_invoke, LLM_confidence_tokens, LLM_confidence_time = 0, 0, 0
    match_count = 0

    pbar = tqdm(total=len(log_lines), desc='Parsing')
    for idx in range(len(log_lines)):
        log = log_lines[idx]
        log = " ".join(log.strip().split())

        if cache.match_log(log):
            match_count += 1
            pass
        else:
            t1 = time.time()
            template, confidences = pretrained_model.parse(
                log, device=devices[0], vtoken=vtoken)
            pretrained_model_time += time.time() - t1

            for i in range(len(confidences)):
                confidences[i] = round(float(confidences[i]), 3)
            flag = check_log_and_template(log, template, confidences)
            if flag != "No":
                topk_examples = select_examples(
                    log, examples)
                iter, max_iter = 0, 3
                while iter < max_iter:
                    if flag == "Complex":
                        t2 = time.time()
                        llm_template, usage = llm.parse_log_complex(
                            log, template, confidences, topk_examples)
                        LLM_time += time.time() - t2
                        LLM_complex_time += time.time() - t2
                        LLM_complex_invoke += 1
                        LLM_complex_tokens += usage['total_tokens']
                    elif flag == "Confidence":
                        t2 = time.time()
                        llm_template, usage = llm.parse_log_confidence(
                            log, template, confidences, topk_examples)
                        LLM_time += time.time() - t2
                        LLM_confidence_time += time.time() - t2
                        LLM_confidence_invoke += 1
                        LLM_confidence_tokens += usage['total_tokens']
                    elif flag == "Keyword":
                        t2 = time.time()
                        llm_template, usage = llm.parse_log_keyword(
                            log, template, confidences, topk_examples)
                        LLM_time += time.time() - t2
                        LLM_keyword_time += time.time() - t2
                        LLM_keyword_invoke += 1
                        LLM_keyword_tokens += usage['total_tokens']

                    LLM_tokens += usage['total_tokens']
                    LLM_invoke += 1
                    llm_template = post_process_template(llm_template)
                    if check_llm_template(log, llm_template):
                        break
                    iter += 1
                if iter < max_iter and \
                        (flag == "Complex" or check_template_match(template, llm_template)):
                    if template != llm_template:
                        logger.info(
                            f"PreTrained Model Parsed Result: {template}")
                        logger.info(
                            f"{flag} LLM Parse Success: {llm_template}")
                    template = llm_template

            similar_templates = cache.find_similar_templates(template)
            if len(similar_templates) > 0:
                merge_flag = False
                for similar_template in similar_templates:
                    m_template, m_example, m_distance = similar_template[
                        0], similar_template[1], similar_template[2]
                    t3 = time.time()
                    flag, merge_template, usage = llm.merge_template(
                        m_template, m_example, template, log, m_distance)
                    LLM_merge_time += time.time() - t3
                    LLM_merge_tokens += usage['total_tokens']
                    LLM_merge_invoke += 1
                    if flag == 'Merge':
                        if check_llm_template(log, merge_template) and check_llm_template(m_example, merge_template):
                            logger.info(
                                f"Merge Template: {merge_template}")
                            cache.update_template(
                                m_template, merge_template, log)
                            merge_flag = True
                        break
                if not merge_flag:
                    cache.insert_template(template, log)
            else:
                cache.insert_template(template, log)

        pbar.update(1)

    LLM_cost_file = f"{common_args.output_dir}/logs/LLM_cost.json"
    LLM_table = {}
    if os.path.exists(LLM_cost_file):
        with open(LLM_cost_file, 'r') as file:
            LLM_table = json.load(file)
    LLM_table[data_args.dataset_name] = {
        'LLM_Complex_Invoke': LLM_complex_invoke,
        'LLM_Complex_Tokens': LLM_complex_tokens,
        'LLM_Complex_Time': round(LLM_complex_time, 3),
        'LLM_Keyword_Invoke': LLM_keyword_invoke,
        'LLM_Keyword_Tokens': LLM_keyword_tokens,
        'LLM_Keyword_Time': round(LLM_keyword_time, 3),
        'LLM_Confidence_Invoke': LLM_confidence_invoke,
        'LLM_Confidence_Tokens': LLM_confidence_tokens,
        'LLM_Confidence_Time': round(LLM_confidence_time, 3),
    }
    with open(LLM_cost_file, 'w') as file:
        json.dump(LLM_table, file)

    parsing_time = time.time() - t0
    cache_time = parsing_time - pretrained_model_time - LLM_time - LLM_merge_time
    logger.info(f"Total time taken: {parsing_time}")
    logger.info(
        f"Total time taken by pretrained model: {pretrained_model_time}")

    templates = []
    for log in log_lines:
        log = " ".join(log.strip().split())
        templates.append(cache.get_template_for_log(log))

    log_df['EventTemplate'] = pd.Series(templates)

    total_count = len(log_lines)
    match_ratio = match_count / total_count

    return log_df, templates, cache_time, pretrained_model_time, LLM_time, LLM_merge_time, LLM_tokens, LLM_merge_tokens, LLM_invoke, LLM_merge_invoke, match_count, total_count, match_ratio


def save_results(common_args, data_args, log_df, templates):
    task_output_dir = f"{common_args.output_dir}/logs"
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    log_df.to_csv(
        f"{task_output_dir}/{data_args.dataset_name}_full.log_structured.csv", index=False)

    counter = Counter(templates)
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    template_df = pd.DataFrame(items, columns=['EventTemplate', 'Occurrence'])
    template_df['EventID'] = [f"E{i + 1}" for i in range(len(template_df))]
    template_df[['EventID', 'EventTemplate', 'Occurrence']].to_csv(
        f"{task_output_dir}/{data_args.dataset_name}_full.log_templates.csv", index=False)


def save_cost(common_args, data_args, training_time, parsing_time, cache_time, pretrained_model_time, LLM_time, LLM_merge_time, LLM_tokens, LLM_merge_tokens, LLM_invoke, LLM_merge_invoke, match_count, total_count, match_ratio):
    time_cost_file = f"{common_args.output_dir}/logs/time_cost.json"
    time_table = {}
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
    time_table[data_args.dataset_name] = {
        'TrainingTime': round(training_time, 3),
        'ParsingTime': round(parsing_time, 3),
        'CacheTime': round(cache_time, 3),
        'PreTrainedModelTime': round(pretrained_model_time, 3),
        'LLM_Time': round(LLM_time, 3),
        'LLM_Merge_Time': round(LLM_merge_time, 3),
        'LLM_Total_Time': round(LLM_time+LLM_merge_time, 3),
        'LLM_Tokens': round(LLM_tokens, 3),
        'LLM_Merge_Tokens': round(LLM_merge_tokens, 3),
        'LLM_Total_Tokens': round(LLM_tokens+LLM_merge_tokens, 3),
        'LLM_Invoke': round(LLM_invoke, 3),
        'LLM_Merge_Invoke': round(LLM_merge_invoke, 3),
        'LLM_Total_Invoke': round(LLM_invoke+LLM_merge_invoke, 3),
        'MatchCount': match_count,
        'TotalCount': total_count,
        'MatchRatio': round(match_ratio, 3),
    }
    with open(time_cost_file, 'w') as file:
        json.dump(time_table, file)


if __name__ == "__main__":
    data_args, model_args, train_args, common_args = get_args()
    if common_args.seed is not None:
        set_seed(common_args.seed)

    sampling(data_args, common_args)

    pretrained_model, data_loader = initialize_and_load_data(
        data_args, model_args)

    trainer, device, devices = setup_training(
        pretrained_model, data_loader, train_args, common_args)

    pretrained_model, training_time = train_model(
        trainer, common_args, data_args)

    t0 = time.time()
    log_df, templates, cache_time, pretrained_model_time, LLM_time, LLM_merge_time, LLM_tokens, LLM_merge_tokens, LLM_invoke, LLM_merge_invoke, match_count, total_count, match_ratio = log_parse(
        data_args, model_args, common_args, pretrained_model, devices, vtoken=data_loader.vtoken)

    save_results(common_args, data_args, log_df, templates)
    parsing_time = time.time() - t0

    save_cost(common_args, data_args, training_time,
              parsing_time, cache_time, pretrained_model_time, LLM_time, LLM_merge_time, LLM_tokens, LLM_merge_tokens, LLM_invoke, LLM_merge_invoke, match_count, total_count, match_ratio)
