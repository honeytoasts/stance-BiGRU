# 3rd-party module
import torch
import pandas as pd

# self-made module
from util import loss
from util import scorer

@torch.no_grad()
def evaluate_function(device, model, config, batch_iterator):
    total_loss = 0.0
    all_label, all_pred, all_pred_p = [], [], []

    # evaluate model
    model.eval()
    for _, target, claim, label in batch_iterator:

        # specify device for data
        target = target.to(device)
        claim = claim.to(device)
        label = label.to(device)

        # get predict label
        predict = model(target, claim)

        # calculate loss
        batch_loss = loss.loss_function(predict=predict,
                                        target=label)

        # sum the batch loss
        total_loss += batch_loss * len(target)

        # get target, label and predict
        all_label.extend(label.cpu().tolist())
        all_pred.extend(
            torch.argmax(predict, axis=1).cpu().tolist())
        all_pred_p.extend(predict.cpu().tolist())

    # evaluate loss
    total_loss /= len(all_label)

    # evaluate f1 score
    target_f1 = scorer.score_function(labels=all_label,
                                      predicts=all_pred)

    return (total_loss.item(), target_f1,
            all_label, all_pred, all_pred_p)