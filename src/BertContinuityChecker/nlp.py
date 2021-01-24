import copy
import re
import codecs
import dataclasses
from typing import List

import chardet

from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
import torch
import torch.nn.functional as F

@dataclasses.dataclass
class SentObject:
    par_number:int
    sent:str

@dataclasses.dataclass
class WindowResult:
    target:str = ""
    do_exit:bool = False

@dataclasses.dataclass
class AnalyzeResult:
    text_data:List[SentObject]
    serial_scores:List[float]
    par_scores:List[float]
    LABEL:str = "\t".join(["段落番号", "データ", "連続度", "同じ段落度"]) + '\n'

    def write_to_tsv(self, path:str):
        # ダメなやつ弾く処理書け
        with codecs.open(path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(self.LABEL)
            f.write("\t".join([
                str(self.text_data[0].par_number),
                self.text_data[0].sent, "", ""
            ])+'\n')
            data = zip(
                self.text_data[1:],
                self.par_scores, self.serial_scores
            )
            for sent, par_s, serial_s in data:
                f.write("\t".join([
                    str(sent.par_number), sent.sent,
                    str(par_s), str(serial_s)
                ])+'\n')

def open_txt(txt_path:str) -> str:
    with open(txt_path, 'rb') as f:
        codec = chardet.detect(f.read())['encoding']

    with open(txt_path, 'r', encoding=codec) as f:
        return f.read()

def remove_short_elm(l:list, min_length:int) -> list:
    return list(filter(lambda x: len(x)>min_length, l))

def line_to_sents(line:str) -> List[str]:
    line = line.strip()
    return remove_short_elm(re.findall(".*?。|.*$", line), 2)

def txt_to_sents(txt:str) -> List[SentObject]:
    lines:List[str] = txt.split("\n")
    ret = []
    for i, l in enumerate(lines):
        sents = line_to_sents(l)
        sents = [SentObject(i, s) for s in sents]
        ret += sents
    return ret

class BertEvaluator:
    def __init__(self, device, serial_model_path, par_model_path):
        self.device = device

        pretrained_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_path, do_lower_case=False
        )
        model = BertForSequenceClassification.from_pretrained(
            pretrained_path, num_labels=2
        )
        self.serial_model = copy.deepcopy(model)
        self.par_model = copy.deepcopy(model)

        self.serial_model.load_state_dict(torch.load(serial_model_path))
        self.serial_model.to(self.device)
        self.par_model.load_state_dict(torch.load(par_model_path))
        self.par_model.to(self.device)

    def evaluate(self, user_input, candidate):
        with torch.no_grad():
            tokenized = self.tokenizer(
                [[user_input, candidate]], return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].to(self.device)
            token_type_ids = tokenized["token_type_ids"].to(self.device)

            result_serial = self.serial_model.forward(
                input_ids, token_type_ids=token_type_ids
            )
            result_serial = F.softmax(result_serial[0], dim=1)
            result_serial = result_serial[0].cpu().numpy().tolist()

            result_par = self.par_model.forward(
                input_ids, token_type_ids=token_type_ids
            )
            result_par = F.softmax(result_par[0], dim=1)
            result_par = result_par[0].cpu().numpy().tolist()

            return result_serial, result_par