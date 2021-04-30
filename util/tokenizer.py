# built-in module
import os
import pickle
from nltk.corpus import stopwords
from nltk import tokenize

# 3rd-party module
from tqdm import tqdm
import pandas as pd

class BaseTokenizer:
    def __init__(self, config):
        # padding token
        self.pad_token = '[pad]'
        self.pad_token_id = 0

        # others
        self.token_to_id = {}
        self.id_to_token = {}
        self.config = config

        self.token_to_id[self.pad_token] = self.pad_token_id
        self.id_to_token[self.pad_token_id] = self.pad_token

    def tokenize(self, sentences):
        # nltk TweetTokenizer
        tokenizer = tokenize.TweetTokenizer()
        sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

        return sentences

    def detokenize(self, sentences):
        return [' '.join(sentence) for sentence in sentences]

    def build_vocabulary(self, sentences):
        # tokenize
        sentences = self.tokenize(sentences)

        # get all tokens
        all_tokens = set()
        for sent in sentences:
            all_tokens |= set(sent)
        all_tokens = sorted(list(all_tokens))

        # build vocabulary
        for idx in range(len(all_tokens)):
            self.token_to_id[all_tokens[idx]] = idx+1  # start from 1, 0 for padding token
            self.id_to_token[idx+1] = all_tokens[idx]

    def convert_tokens_to_ids(self, sentences):
        result = []

        for sentence in sentences:
            ids = []
            for token in sentence:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    ids.append(self.pad_token_id)
            result.append(ids)

        return result

    def convert_ids_to_tokens(self, sentences):
        result = []

        for sentence in sentences:
            tokens = []
            for idx in sentence:
                if idx in self.id_to_token:
                    tokens.append(self.id_to_token[idx])
                else:
                    raise ValueError(f'idx {idx} not in the dictionary')
            result.append(tokens)

        return result

    def encode(self, sentences):
        # convert tokens to ids
        sentences = self.convert_tokens_to_ids(self.tokenize(sentences))

        # get max length of sentence
        max_sent_len = max([len(sent) for sent in sentences])

        # padding
        for i in range(len(sentences)):
            sentence = sentences[i]

            # padding
            pad_count = max_sent_len - len(sentence)
            sentence.extend([self.pad_token_id] * pad_count)

            sentences[i] = sentence

        return sentences

    def decode(self, sentences):
        sentences = self.detokenize(self.convert_ids_to_tokens(sentences))

        return sentences

    def load(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            tokenizer = pickle.load(f)
            self.token_to_id = tokenizer.token_to_id
            self.id_to_token = tokenizer.id_to_token

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)