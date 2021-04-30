# built-in modules
import argparse
import os
import json

class BaseConfig:
    def __init__(self):
        # construct argparser
        parser = argparse.ArgumentParser(
            description='Train the stance-GCN model'
        )

        # add argument to argparser

        # experiment_no
        parser.add_argument('--experiment_no',
                            default='1',
                            type=str)

        # dataset
        parser.add_argument('--embedding',
                            default='wikipedia',
                            type=str)
        parser.add_argument('--output_dim',
                            default=3,
                            type=int)

        # hyperparameter
        parser.add_argument('--embedding_dim',
                            default=300,
                            type=int)
        parser.add_argument('--hidden_dim',
                            default=256,
                            type=int)

        parser.add_argument('--num_rnn_layers',
                            default=1,
                            type=int)
        parser.add_argument('--num_linear_layers',
                            default=1,
                            type=int)
        parser.add_argument('--rnn_dropout',
                            default=0.3,
                            type=float)
        parser.add_argument('--linear_dropout',
                            default=0.5,
                            type=float)

        parser.add_argument('--learning_rate',
                            default=1e-3,
                            type=float)
        parser.add_argument('--weight_decay',
                            default=0.0,
                            type=float)

        # other
        parser.add_argument('--random_seed',
                            default=77,
                            type=int)
        parser.add_argument('--kfold',
                            default=5,
                            type=int)
        parser.add_argument('--epoch',
                            default=50,
                            type=int)
        parser.add_argument('--batch_size',
                            default=16,
                            type=int)

        self.parser = parser
        self.config = None

    def get_config(self):
        self.config = self.parser.parse_args()

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')

        # save config to json format
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False)

    def load(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        # load config file
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.config = argparse.Namespace(**config)