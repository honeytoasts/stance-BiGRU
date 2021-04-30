# 3rd-party module
import pandas as pd
from sklearn import metrics

def score_function(labels, predicts):
    consider_labels = [0, 1]  # only consider "favor (0)" and "against (1)"

    target_f1 = metrics.f1_score(labels,
                                 predicts,
                                 average='macro',
                                 labels=consider_labels,
                                 zero_division=0)

    return target_f1