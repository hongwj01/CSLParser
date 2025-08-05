from datasets import load_dataset
import copy
import re
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification


class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name]
                  for feature in features] if label_name in features[0].keys() else None
        ori_labels = [feature['ori_labels']
                      for feature in features] if 'ori_labels' in features[0].keys() else None
        max_length = max([len(x['input_ids']) for x in features])
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=min(max_length, 256),
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] *
                               (sequence_length - len(label)) for label in labels]
            batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                                   ori_labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] *
                               (sequence_length - len(label)) + label for label in labels]
            batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                                   ori_labels]

        batch = {k: torch.tensor(v, dtype=torch.int64)
                 for k, v in batch.items()}
        return batch


class DataLoaderForPromptTuning():
    def __init__(self, config):
        self.config = config
        self.vtoken = "<*>"
        self.delimiters = r"([ |\(|\)|\[|\]|\{|\})])"
        self.load_data()

    def load_data(self):
        data_files = {}
        if self.config.train_file is not None:
            data_files["train"] = [self.config.train_file]
        if self.config.validation_file is not None:
            data_files["validation"] = self.config.validation_file

        self.raw_datasets = load_dataset("json", data_files=data_files)

        if self.raw_datasets["train"] is not None:
            column_names = self.raw_datasets["train"].column_names
        else:
            column_names = self.raw_datasets["validation"].column_names

        if self.config.text_column_name is not None:
            text_column_name = self.config.text_column_name
        else:
            text_column_name = column_names[0]

        if self.config.label_column_name is not None:
            label_column_name = self.config.label_column_name
        else:
            label_column_name = column_names[1]
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    def initialize(self, tokenizer):
        self.tokenizer = tokenizer
        self.ori_label_token_map = {
            self.vtoken: []
        }

        sorted_add_tokens = sorted(
            list(self.ori_label_token_map.keys()), key=lambda x: len(x), reverse=True)
        self.tokenizer.add_tokens(sorted_add_tokens)

        self.label_list = list(self.ori_label_token_map.keys())
        self.label_list += 'origin'
        self.label_to_id = {'origin': 0}
        for label in self.label_list:
            if label != 'origin':
                self.label_to_id[label] = len(self.label_to_id)

        self.id_to_label = {id: label for label,
                            id in self.label_to_id.items()}
        self.label_token_map = {
            item: item for item in self.ori_label_token_map}
        self.label_token_to_id = {label: tokenizer.convert_tokens_to_ids(label_token) for label, label_token in
                                  self.label_token_map.items()}
        self.label_token_id_to_label = {
            idx: label for label, idx in self.label_token_to_id.items()}

        return self.tokenizer

    def get_template_regex(self, template):
        if "<*>" not in template:
            return None
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace(r"\<\*\>", "(.*?)") + "$"
        return template_regex

    def tokenize(self):
        label_words = []
        keywords = []

        def tokenize_and_align_labels(examples):
            target_tokens = []
            tag_labels = []
            input_ids = []
            attention_masks = []
            for i, (log, label) in enumerate(zip(examples[self.text_column_name], examples[self.label_column_name])):
                log = " ".join(log.strip().split())
                label = " ".join(label.strip().split())
                template_regex = self.get_template_regex(label)
                if template_regex is None:
                    input_tokens, label_tokens = [log], ['origin']
                else:
                    match = next(re.finditer(template_regex, log))
                    input_tokens = []
                    label_tokens = []
                    cur_position = 0
                    for idx in range(match.lastindex):
                        start, end = match.span(idx + 1)
                        if start > cur_position:
                            input_tokens.append(
                                log[cur_position:start].rstrip())
                            label_tokens.append('origin')
                        input_tokens.append(log[start:end])
                        if start > 0 and log[start - 1] == " ":
                            input_tokens[-1] = " " + input_tokens[-1]
                        label_tokens.append(self.vtoken)
                        cur_position = end

                    if cur_position < len(log):
                        input_tokens.append(
                            log[cur_position:len(log)].rstrip())
                        label_tokens.append('origin')

                refined_tokens = []
                refined_labels = []
                for (t, l) in zip(input_tokens, label_tokens):
                    if len(t) == 0:
                        continue
                    t = re.split(self.delimiters, t)
                    t = [x for x in t if len(x) > 0]
                    sub_tokens = []
                    if t[0] != " ":
                        sub_tokens.append(t[0])
                    for i in range(1, len(t)):
                        if t[i] == " ":
                            continue
                        if t[i - 1] == " ":
                            sub_tokens.append(" " + t[i])
                        else:
                            sub_tokens.append(t[i])
                    refined_tokens.extend(sub_tokens)
                    refined_labels.extend([l] * len(sub_tokens))

                input_id = []
                labels = []
                target_token = []
                for (input_token, label_token) in zip(refined_tokens, refined_labels):
                    token_ids = self.tokenizer.encode(
                        input_token, add_special_tokens=False)
                    input_id.extend(token_ids)
                    if label_token != self.vtoken:
                        target_token.extend(token_ids)
                        keywords.extend(token_ids)
                    else:
                        target_token.extend(
                            [self.label_token_to_id[label_token]] * len(token_ids))
                        label_words.extend(token_ids)
                    labels.extend([self.label_to_id[label_token]]
                                  * len(token_ids))

                input_id = [self.tokenizer.cls_token_id] + \
                    input_id + [self.tokenizer.sep_token_id]
                target_token = [self.tokenizer.bos_token_id] + \
                    target_token + [self.tokenizer.eos_token_id]
                labels = [-100] + labels + [-100]
                attention_mask = [1] * len(input_id)
                input_ids.append(input_id)
                target_tokens.append(target_token)
                tag_labels.append(labels)
                attention_masks.append(attention_mask)

            return {
                "input_ids": input_ids,
                "labels": target_tokens,
                "ori_labels": tag_labels,
                "attention_mask": attention_masks
            }

        self.processed_raw_datasets = {}
        self.processed_raw_datasets['train'] = self.raw_datasets['train'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
            desc="Running tokenizer on train dataset"
        )

        self.keywords = list(set(keywords))
        self.label_words = list(set(label_words))
        self.label_words = [x for x in label_words if x not in self.keywords]

        if 'validation' in self.raw_datasets:
            self.processed_raw_datasets['validation'] = self.raw_datasets['validation'].map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=self.raw_datasets["validation"].column_names,
                desc="Running tokenizer on test dataset",
                num_proc=4
            )

    def build_dataloaders(self, per_device_train_batch_size, per_device_eval_batch_size):
        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer, pad_to_multiple_of=None)
        self.train_loader = DataLoader(
            self.processed_raw_datasets['train'],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size
        )
        if 'validation' in self.processed_raw_datasets:
            self.val_loader = DataLoader(
                self.processed_raw_datasets['validation'],
                collate_fn=data_collator,
                batch_size=per_device_eval_batch_size
            )
        else:
            self.val_loader = None

    def size(self):
        return len(self.raw_datasets['train'])

    def get_train_dataloader(self):
        return self.train_loader

    def get_val_dataloader(self):
        return self.val_loader
