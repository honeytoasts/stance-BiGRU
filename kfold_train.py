# built-in module
import os
import random

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

# self-made module
import util

# prevent warning
pd.options.mode.chained_assignment = None

def train_and_test(target, abbr_target, config):
    print(f'\ntrain and test for {abbr_target}')

    # define save path
    save_path = f'model/{config.experiment_no}/{abbr_target}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise FileExistsError(f'experiment {config.experiment_no} is exist')

    # initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # set random seed and ensure deterministic
    os.environ['PYTHONHASHSEED'] = str(config.random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # initialize tokenizer
    tokenizer = util.tokenizer.BaseTokenizer(config)

    # initialize embedding
    embedding = util.embedding.BaseEmbedding(
        embedding=config.embedding,
        embedding_dim=config.embedding_dim)

    # get data
    data = util.data.SemevalData(training=True,
                                 target=target,
                                 config=config,
                                 tokenizer=tokenizer,
                                 embedding=embedding)
    data_df, test_df = data.train_df, data.test_df

    # initialize
    all_valid_target_f1, all_test_pred_p = [], []

    # initialize tensorboard
    writer = SummaryWriter(f'tensorboard/exp-{config.experiment_no}')

    # kfold training
    kfold = StratifiedKFold(n_splits=config.kfold,
                            shuffle=True,
                            random_state=config.random_seed)

    for fold, (train_index, valid_index) in enumerate(
        kfold.split(data_df['target'], data_df['label']), start=1):
        print(f'{fold}-fold')

        # initialize
        best_valid_target_f1, best_test_pred_p = None, None

        # get training and validation data
        train_df = data_df.iloc[train_index]
        valid_df = data_df.iloc[valid_index]

        # create dataset
        train_dataset = util.data.Dataset(target_name=train_df['target'],
                                          target=train_df['target_encode'],
                                          claim=train_df['claim_encode'],
                                          labels=train_df['label_encode'])
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

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     collate_fn=collate_fn)

        # construct model
        model = util.model.BaseModel(device=device,
                                     config=config,
                                     num_embeddings=embedding.get_num_embeddings(),
                                     padding_idx=tokenizer.pad_token_id,
                                     embedding_weight=embedding.vector)
        model = model.to(device)

        # construct optimizer
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)

        # training model
        model.zero_grad()

        for epoch in range(int(config.epoch)):
            print()
            model.train()
            train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                                  desc=f'epoch {epoch}', position=0)

            for _, target, claim, label in train_iterator:

                # specify device for data
                target = target.to(device)
                claim = claim.to(device)
                label = label.to(device)

                # get prediction
                predict = model(target, claim)

                # clean up gradient
                optimizer.zero_grad()

                # get loss
                loss = util.loss.loss_function(predict, label)

                # backward pass
                loss.backward()

                # gradient decent
                optimizer.step()

            # evaluate model
            train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                                  desc='evaluate training data', position=0)
            train_loss, train_target_f1, _, _, _ = (
                util.evaluate.evaluate_function(device=device,
                                                model=model,
                                                config=config,
                                                batch_iterator=train_iterator))

            valid_iterator = tqdm(valid_dataloader, total=len(valid_dataloader),
                                  desc='evaluate validation data', position=0)
            valid_loss, valid_target_f1, _, _, _ = (
                util.evaluate.evaluate_function(device=device,
                                                model=model,
                                                config=config,
                                                batch_iterator=valid_iterator))

            test_iterator = tqdm(test_dataloader, total=len(test_dataloader),
                                 desc='evaluate test data', position=0)
            _, _, test_label, _, test_pred_p = (
                util.evaluate.evaluate_function(device=device,
                                                model=model,
                                                config=config,
                                                batch_iterator=test_iterator))

            # print loss and score
            print(f'train loss: {round(train_loss, 5)}, '
                  f'train f1: {round(train_target_f1, 5)}\n'
                  f'valid loss: {round(valid_loss, 5)}, '
                  f'valid f1: {round(valid_target_f1, 5)}')

            # save model
            if best_valid_target_f1 is None or (
                valid_target_f1 > best_valid_target_f1):

                best_valid_target_f1 = valid_target_f1
                best_test_pred_p = test_pred_p

                model_path = f'{save_path}/{fold}-fold'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(model.state_dict(), f'{model_path}/model_{epoch}.ckpt')

            # write loss and f1 to tensorboard
            writer.add_scalars(f'Loss/train_{abbr_target}',
                               {f'{fold}-fold': train_loss}, epoch)
            writer.add_scalars(f'Loss/valid_{abbr_target}',
                               {f'{fold}-fold': valid_loss}, epoch)

            writer.add_scalars(f'F1/train_{abbr_target}',
                               {f'{fold}-fold': train_target_f1}, epoch)
            writer.add_scalars(f'F1/valid_{abbr_target}',
                               {f'{fold}-fold': valid_target_f1}, epoch)

        # save each fold f1 score and test pred prob.
        all_valid_target_f1.append(best_valid_target_f1)
        all_test_pred_p.append(best_test_pred_p)

        # release GPU memory
        torch.cuda.empty_cache()

    # save tokenizer and embedding
    tokenizer.save(f'{save_path}/tokenizer.pickle')
    embedding.save(f'{save_path}/embedding.pickle')

    # calucalte average valid f1 and test f1
    valid_target_f1 = sum(all_valid_target_f1)/len(all_valid_target_f1)

    test_pred_p = torch.tensor(all_test_pred_p)
    test_pred = test_pred_p.sum(dim=0).argmax(dim=1).tolist()
    test_target_f1 = util.scorer.score_function(labels=test_label,
                                                predicts=test_pred)

    return valid_target_f1, test_target_f1, test_label, test_pred

def main():
    # initialize config
    config_cls = util.config.BaseConfig()
    config_cls.get_config()
    config = config_cls.config

    # define targets
    targets = ['Atheism', 'Climate Change is a Real Concern',
               'Feminist Movement', 'Hillary Clinton',
               'Legalization of Abortion']
    abbr_targets = ['atheism', 'climate', 'feminism', 'hillary', 'abortion']

    # train and evaluate for each target
    all_valid_target_f1, all_test_target_f1 = [], []
    all_test_label, all_test_pred = [], []

    for target, abbr_target in zip(targets, abbr_targets):
        valid_target_f1, test_target_f1, test_label, test_pred = (
            train_and_test(target=target,
                           abbr_target=abbr_target,
                           config=config))

        all_valid_target_f1.append(valid_target_f1)
        all_test_target_f1.append(test_target_f1)
        all_test_label.extend(test_label)
        all_test_pred.extend(test_pred)

    # save config
    config_cls.save(f'model/{config.experiment_no}/config.json')

    # save hyperparameters
    writer = SummaryWriter(f'tensorboard/exp-{config.experiment_no}')
    writer.add_hparams(
        {str(key): str(value) for key, value in config.__dict__.items()},
        metric_dict={})
    writer.close()

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