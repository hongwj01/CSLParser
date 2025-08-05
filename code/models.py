from transformers import AutoTokenizer, RobertaForMaskedLM, AutoModelForMaskedLM
import re
import torch
from torch import nn
import torch.nn.functional as F
from post_process import correct_single_template

delimiters = "([ |\(|\)|\[|\]|\{|\})])"


def calculate_confidence(tokenizer, c, t, probs, vtoken="virtual-param"):
    vtoken = tokenizer.convert_tokens_to_ids(vtoken)
    tokens = tokenizer.convert_ids_to_tokens(c)

    var_probs = []
    res = [" "]
    for i in range(1, len(c)):
        if c[i] == tokenizer.sep_token_id:
            break
        if t[i] < vtoken:
            res.append(tokens[i])
        else:
            if "Ġ" in tokens[i]:
                res.append("Ġ<*>")
                var_probs.append(probs[0, i].max().item())
            elif "<*>" not in res[-1]:
                res.append("<*>")
                var_probs.append(probs[0, i].max().item())

    return var_probs


def map_template(tokenizer, c, t, vtoken="virtual-param"):
    vtoken = tokenizer.convert_tokens_to_ids(vtoken)
    tokens = tokenizer.convert_ids_to_tokens(c)
    res = [" "]
    for i in range(1, len(c)):
        if c[i] == tokenizer.sep_token_id:
            break
        if t[i] < vtoken:
            res.append(tokens[i])
        else:
            if "Ġ" in tokens[i]:
                res.append("Ġ<*>")
            elif "<*>" not in res[-1]:
                res.append("<*>")
    r = "".join(res)
    r = r.replace("Ġ", " ")
    return r.strip()


class PreTrainedModel(nn.Module):
    def __init__(self,
                 model_path,
                 num_label_tokens: int = 1,
                 vtoken='virtual-param',
                 mode='train'
                 ):
        super().__init__()
        self.model_path = model_path
        self.num_label_tokens = num_label_tokens
        print(self.model_path)

        if self.model_path == './models/roberta-base':
            self.model = RobertaForMaskedLM.from_pretrained(self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings - 2
        self.vtoken = vtoken

        if mode == 'train':
            self.add_label_token(self.num_label_tokens)

    def forward(self, batch):
        outputs = self.model(**batch, output_hidden_states=True)
        loss = outputs.loss
        return loss

    def add_label_token(self, num_tokens: int):
        if self.model_path == './models/roberta-base':
            crr_tokens, _ = self.model.roberta.embeddings.word_embeddings.weight.shape
        self.model.resize_token_embeddings(crr_tokens + num_tokens)

    def named_parameters(self):
        return self.model.named_parameters()

    def eval(self):
        self.model.eval()

    def parse(self, log_line, device="cpu", vtoken="virtual-param"):
        def tokenize(log_line, max_length=256):
            log_tokens = re.split(delimiters, log_line)
            log_tokens = [token for token in log_tokens if len(token) > 0]
            refined_tokens = []
            if log_tokens[0] != " ":
                refined_tokens.append(log_tokens[0])
            for i in range(1, len(log_tokens)):
                if log_tokens[i] == " ":
                    continue
                if log_tokens[i - 1] == " ":
                    refined_tokens.append(" " + log_tokens[i])
                else:
                    refined_tokens.append(log_tokens[i])
            token_ids = []
            for token in refined_tokens:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                token_ids.extend(ids)
            token_ids = token_ids[:max_length - 2]
            token_ids = [self.tokenizer.bos_token_id] + \
                token_ids + [self.tokenizer.eos_token_id]
            return {
                'input_ids': torch.tensor([token_ids], dtype=torch.int64),
                'attention_mask': torch.tensor([[1] * len(token_ids)], dtype=torch.int64)
            }

        tokenized_input = tokenize(log_line)
        tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
        with torch.no_grad():
            outputs = self.model(**tokenized_input, output_hidden_states=True)
        logits = outputs.logits.argmax(dim=-1)
        probs = F.softmax(outputs.logits.float(), dim=-1)
        confidences = calculate_confidence(
            self.tokenizer, tokenized_input['input_ids'][0], logits[0], probs, vtoken=vtoken)
        logits = logits.detach().cpu().clone().tolist()
        template = map_template(
            self.tokenizer, tokenized_input['input_ids'][0], logits[0], vtoken=vtoken)
        return correct_single_template(template), confidences

    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_pretrained(self, input_dir):
        self.model = self.model.from_pretrained(input_dir)
