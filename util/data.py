# built-in module
import argparse
import unicodedata
import re
import gc

# 3rd-party module
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# self-made module
from . import config
from . import tokenizer
from . import embedding

class SemevalData:
    def __init__(self,
                 training: bool,
                 target: str,
                 config: argparse.Namespace,
                 tokenizer: tokenizer.BaseTokenizer,
                 embedding: embedding.BaseEmbedding):

        # load data
        file_paths = [
            'data/semeval2016/semeval2016-task6-trialdata.txt',
            'data/semeval2016/semeval2016-task6-trainingdata.txt',
            'data/semeval2016/SemEval2016-Task6-subtaskA-testdata-gold.txt']

        data_df = pd.DataFrame()
        for file_path in file_paths:
            temp_df = pd.read_csv(file_path, encoding='windows-1252', delimiter='\t')
            data_df = pd.concat([data_df, temp_df])

        data_df.columns = ['ID', 'target_orig', 'claim_orig', 'label']

        self.data = data_df

        # get specific target data
        if target != 'all':
            self.data = self.data[self.data['target_orig'] == target]

        # data preprocessing
        self.preprocessing()

        # if training then init tokenizer and embedding
        if training:
            # build vocabulary
            all_sentences = []
            all_sentences.extend(self.data['target'].tolist())
            all_sentences.extend(self.data['claim'].tolist())

            tokenizer.build_vocabulary(all_sentences)

            # get embeddings
            embedding.load_embedding(id_to_token=tokenizer.id_to_token)

        # content encode
        self.data['target_encode'] = tokenizer.encode(
            sentences=self.data['target'].tolist())
        self.data['claim_encode'] = tokenizer.encode(
            sentences=self.data['claim'].tolist())

        # label encode
        stance_label = {'favor': 0, 'against': 1, 'none': 2}
        self.data['label_encode'] = self.data['label'].apply(
            lambda label: stance_label[label])

        self.train_df = self.data[self.data['ID'] <= 10000]
        self.test_df = self.data[self.data['ID'] >= 10000]

        # reset index
        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

    def preprocessing(self):
        # encoding normalize
        normalize_func = (
            lambda text: unicodedata.normalize('NFKC', str(text)))

        self.data['target'] = self.data['target_orig'].apply(normalize_func)
        self.data['claim'] = self.data['claim_orig'].apply(normalize_func)
        self.data['label'] = self.data['label'].apply(normalize_func)

        # tweet preprocessing
        self.data['claim'] = self.data['claim'].apply(
            self.tweet_preprocessing)

        # change to lower case
        lower_func = lambda text: text.lower().strip()

        self.data['target'] = self.data['target'].apply(lower_func)
        self.data['claim'] = self.data['claim'].apply(lower_func)
        self.data['label'] = self.data['label'].apply(lower_func)

    def tweet_preprocessing(self, text):
        # reference: https://github.com/zhouyiwei/tsd/blob/master/utils.py

        text = text.replace('#SemST', '').strip()

        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", text)
        text = re.sub(r"@\w+", "<USER>", text)
        text = re.sub(r"[8:=;]['`\-]?[)d]+|[)d]+['`\-]?[8:=;]", "<SMILE>", text)
        text = re.sub(r"[8:=;]['`\-]?p+", "<LOLFACE>", text)
        text = re.sub(r"[8:=;]['`\-]?\(+|\)+['`\-]?[8:=;]", "<SADFACE>", text)
        text = re.sub(r"[8:=;]['`\-]?[\/|l*]", "<NEUTRALFACE>", text)
        text = re.sub(r"<3","<HEART>", text)
        text = re.sub(r"/"," / ", text)
        text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
        p = re.compile(r"#\S+")
        text = p.sub(lambda s: "<HASHTAG> "+s.group()[1:]+" <ALLCAPS>"
                     if s.group()[1:].isupper()
                     else " ".join(["<HASHTAG>"]+re.split(r"([A-Z][^A-Z]*)",
                                   s.group()[1:])),text)
        text = re.sub(r"([!?.]){2,}", r"\1 <REPEAT>", text)
        text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <ELONG>", text)

        return text

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 target_name: pd.Series,
                 target: pd.Series,
                 claim: pd.Series,
                 labels: pd.Series):

        self.target_name = target_name.reset_index(drop=True)
        self.target = [ids for ids in target]
        self.claim = [ids for ids in claim]
        self.label = [label for label in labels]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (self.target_name[index],
                self.target[index],
                self.claim[index],
                self.label[index])

# reference: https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/3
class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        target_name = [data[0] for data in batch]
        target = [torch.LongTensor(data[1]) for data in batch]
        claim = [torch.LongTensor(data[2]) for data in batch]
        label = torch.LongTensor([data[3] for data in batch])

        # pad target to fixed length with padding token id
        target = torch.nn.utils.rnn.pad_sequence(target,
                                                 batch_first=True,
                                                 padding_value=self.pad_token_id)

        # pad claim to fixed length with padding token id
        claim = torch.nn.utils.rnn.pad_sequence(claim,
                                                batch_first=True,
                                                padding_value=self.pad_token_id)

        return target_name, target, claim, label