# built-in module
import os
import pickle

# 3rd-party module
import torch
from tqdm import tqdm

class BaseEmbedding:
    def __init__(self, embedding='twitter', embedding_dim=300):
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.vector = torch.Tensor()

        # padding token
        self.add_embedding(torch.zeros(self.embedding_dim))

    def get_num_embeddings(self):
        return self.vector.shape[0]

    def add_embedding(self, vector=None):
        if vector is not None:
            vector = vector.unsqueeze(0)
        else:
            vector = torch.empty(1, self.embedding_dim)
            torch.nn.init.normal_(vector, mean=0, std=1)

        self.vector = torch.cat([self.vector, vector], dim=0)

    def load_embedding(self, id_to_token):
        word_embeddings = {}
        vectors = []

        # embedding path
        if self.embedding == 'twitter':
            embedding_path = f'data/embedding/glove/glove.twitter.27B.{self.embedding_dim}d.txt'
        elif self.embedding == 'wikipedia':
            embedding_path = f'data/embedding/glove/glove.6B.{self.embedding_dim}d.txt'

        # get number of rows in file
        with open(embedding_path) as f:
            file_len = len(f.readlines())

        # get all embedding in file
        with open(embedding_path) as f:
            firstrow = f.readline()

            # if first row not the header, seek to 0
            if len(firstrow.strip().split()) >= self.embedding_dim:
                f.seek(0)

            for row in tqdm(f, desc='load embedding', total=file_len):
                # get token and embedding
                row = row.strip().split()
                word_embeddings[row[0]] = [float(v) for v in row[1:]]

        # get word vocabulary embedding
        for _, token in id_to_token.items():
            if token in word_embeddings:
                vectors.append(word_embeddings[token])
            else:
                vectors.append([0.0 for _ in range(self.embedding_dim)])

        vectors = torch.Tensor(vectors)
        self.vector = torch.cat([self.vector, vectors], dim=0)

        # check embedding dimension
        assert self.embedding_dim == self.vector.shape[1]

    def load(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            embedding = pickle.load(f)
            self.vector = embedding.vector

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)