# built-in module
import os
import random
import argparse

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold

# self-made module
import util

# prevent warning
pd.options.mode.chained_assignment = None

def get_config():
    # construct parser
    parser = argparse.ArgumentParser(
        description='Evaluate model for stance-BiGRU'
    )

    # experiment_no
    parser.add_argument('--experiment_no',
                        default='1',
                        type=str)

    return parser.parse_args()

def test(target, abbr_target, experiment_no):
    # define model path
    model_path = f'model/{experiment_no}/{abbr_target}'

    # initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # load config
    config_cls = util.config.BaseConfig()
    config_cls.load(file_path=f'model/{experiment_no}/config.json')
    config = config_cls.config

    # load tokenizer
    tokenizer = util.tokenizer.BaseTokenizer(config)
    tokenizer.load(file_path=f'{model_path}/tokenizer.pickle')

    # load embedding
    embedding = util.embedding.BaseEmbedding(
        embedding=config.embedding,
        embedding_dim=config.embedding_dim)
    embedding.load(file_path=f'{model_path}/embedding.pickle')

    # get data
    data = util.data.SemevalData(training=False,
                                 target=target,
                                 config=config,
                                 tokenizer=tokenizer,
                                 embedding=embedding)
    data_df, test_df = data.train_df, data.test_df

    # initialize
    all_valid_target_f1, all_test_pred_p = [], []

    # get the last model of each fold
    model_list = []
    for fold in range(1, 6):
        models = os.listdir(f'{model_path}/{fold}-fold')
        models.sort()
        models.sort(key=len)
        model_list.append(f'{fold}-fold/{models[-1]}')
    
    # kfold evaluate
    kfold = StratifiedKFold(n_splits=config.kfold,
                            shuffle=True,
                            random_state=config.random_seed)

    for fold, (_, valid_index) in enumerate(
        kfold.split(data_df['target'], data_df['label']), start=0):

        # get validation data
        valid_df = data_df.iloc[valid_index]

        # create dataset
        valid_dataset = util.data.Dataset(target_name=valid_df['target'],
                                          target=valid_df['target_encode'],
                                          claim=valid_df['claim_encode'],
                                          labels=valid_df['label_encode'])
        test_dataset = util.data.Dataset(target_name=test_df['target'],
                                         target=test_df['target_encode'],
                                         claim=test_df['claim_encode'],
                                         labels=test_df['label_encode'])
        # create dataloader
        collate_fn = util.data.Collator(tokenizer.pad_token_id)

        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     collate_fn=collate_fn)

        # load model
        model = util.model.BaseModel(device=device,
                                     config=config,
                                     num_embeddings=embedding.get_num_embeddings(),
                                     padding_idx=tokenizer.pad_token_id,
                                     embedding_weight=embedding.vector)
        model.load_state_dict(
            torch.load(f'{model_path}/{model_list[int(fold)]}'))
        model = model.to(device)

        # evaluate
        _, valid_target_f1, _, _, _ = (
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=valid_dataloader))

        _, _, test_label, _, test_pred_p = (
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=test_dataloader))

        # store f1 score and predict prob.
        all_valid_target_f1.append(valid_target_f1)
        all_test_pred_p.append(test_pred_p)

        # release GPU memory
        torch.cuda.empty_cache()

    # calucalte average valid f1 and test f1
    valid_target_f1 = sum(all_valid_target_f1)/len(all_valid_target_f1)

    test_pred_p = torch.tensor(all_test_pred_p)
    test_pred = test_pred_p.sum(dim=0).argmax(dim=1).tolist()
    test_target_f1 = util.scorer.score_function(labels=test_label,
                                                predicts=test_pred)

    return valid_target_f1, test_target_f1, test_label, test_pred

def main():
    # get config
    config = get_config()
    experiment_no = config.experiment_no

    # define targets
    targets = ['Atheism', 'Climate Change is a Real Concern',
               'Feminist Movement', 'Hillary Clinton',
               'Legalization of Abortion']
    abbr_targets = ['atheism', 'climate', 'feminism', 'hillary', 'abortion']

    # evaluate each target f1 score
    all_valid_target_f1, all_test_target_f1 = [], []
    all_test_label, all_test_pred = [], []

    for target, abbr_target in zip(targets, abbr_targets):
        valid_target_f1, test_target_f1, test_label, test_pred = (
            test(target=target,
                 abbr_target=abbr_target,
                 experiment_no=experiment_no))

        all_valid_target_f1.append(valid_target_f1)
        all_test_target_f1.append(test_target_f1)
        all_test_label.extend(test_label)
        all_test_pred.extend(test_pred)

    # calculate micro and macro f1 score
    avg_target_f1 = sum(all_valid_target_f1)/len(all_valid_target_f1)
    macro_f1 = sum(all_test_target_f1)/len(all_test_target_f1)
    micro_f1 = util.scorer.score_function(labels=all_test_label,
                                          predicts=all_test_pred)

    # print result
    for i, abbr_target in enumerate(abbr_targets):
        print(f'\nResult for {abbr_target}:\n'
              f'valid f1: {round(all_valid_target_f1[i], 5)}\n'
              f'target f1: {round(all_test_target_f1[i], 5)}')    

    print(f'\nFinal results:\n'
          f'valid f1: {round(avg_target_f1, 5)}\n'
          f'macro f1: {round(macro_f1, 5)}\n'
          f'micro f1: {round(micro_f1, 5)}')

if __name__ == '__main__':
    main()